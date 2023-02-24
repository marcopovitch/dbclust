#!/usr/bin/env python
import os
import sys
import logging
from math import pow, sqrt
import numpy as np
import pandas as pd
from itertools import product
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import functools
from itertools import filterfalse
import dask.bag as db

# from dask.cache import Cache
from obspy import Catalog
from obspy.core.event import Event
from obspy.core.event.base import WaveformStreamID
from obspy.core.event.origin import Pick
from obspy.geodetics import gps2dist_azimuth

try:
    from phase import import_phases, import_eqt_phases
except:
    from dbclust.phase import import_phases, import_eqt_phases


# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("clusterize")
logger.setLevel(logging.DEBUG)


@functools.lru_cache(maxsize=None)
def compute_tt(p1, p2, vmean):
    # lru_cache doesn't work with multiprocessing/dask/etc.
    distance, az, baz = gps2dist_azimuth(
        p1.coord["latitude"],
        p1.coord["longitude"],
        p2.coord["latitude"],
        p2.coord["longitude"],
    )
    # distance in meters, convert it to km
    distance = distance / 1000.0
    dd = distance / vmean
    dt = p1.time - p2.time
    tt = sqrt(pow(dt, 2) + pow(dd, 2))
    return tt


def check_phase_take_over(p1, p2):
    if p1.network == p2.network and p1.station == p2.station and p1.phase == p2.phase:
        if p1.proba > p2.proba:
            # print(f"TAKEOVER: {p1} {p2}")
            return "takeover"
        else:
            # print(f"DROP: {p1} {p2}")
            return "drop"
    else:
        # print(f"INSERT: {p1} {p2}")
        return "insert"


class Clusterize(object):
    def __init__(
        self,
        phases,
        max_search_dist,
        min_size,
        average_velocity,
        min_station_count=None,
        P_uncertainty=0.1,
        S_uncertainty=0.2,
        tt_maxtrix_fname="tt_matrix.npy",
        tt_matrix_load=False,
        tt_matrix_save=False,
    ):
        # clusters is a list of cluster :
        # ie. [ [phases, label], ... ]
        # noise is [ phases, -1]
        self.clusters = []
        self.n_clusters = 0
        self.noise = []
        self.n_noise = 0

        # clustering parameters
        self.max_search_dist = max_search_dist
        self.min_size = min_size
        self.average_velocity = average_velocity

        # pick filtering parameters
        self.min_station_count = min_station_count
        self.P_uncertainty = P_uncertainty
        self.S_uncertainty = S_uncertainty

        # tt_matrix load/save parameters
        self.tt_maxtrix_fname = tt_maxtrix_fname
        self.tt_matrix_load = tt_matrix_load
        self.tt_matrix_save = tt_matrix_save

        logger.info(
            f"Starting Clustering (nbphases={len(phases)}, min_size={min_size})."
        )
        if len(phases) < min_size:
            logger.info("Too few picks !")
            # add noise points
            self.clusters = []
            self.n_clusters = 0
            self.noise = [phases, -1]
            self.n_noise = len(phases)
            return

        logger.info("Computing TT matrix.")
        if tt_matrix_load and tt_maxtrix_fname:
            logger.info(f"Loading tt_matrix {tt_maxtrix_fname}.")
            try:
                pseudo_tt = np.load(tt_maxtrix_fname)
            except Exception as e:
                logger.error(e)
                logger.error("Check your config file !")
                sys.exit()
        else:
            # sequential computation
            pseudo_tt = self.compute_tt_matrix(phases, average_velocity)
            # pseudo_tt = self.numpy_compute_tt_matrix(phases, average_velocity)
            # // computation using dask bag: slower for small cluster
            # pseudo_tt = self.dask_compute_tt_matrix(phases, average_velocity)
            logger.info(f"TT maxtrix: {compute_tt.cache_info()}")
            compute_tt.cache_clear()

        if tt_maxtrix_fname and tt_matrix_save:
            logger.info(f"Saving tt_matrix {tt_maxtrix_fname}.")
            np.save(tt_maxtrix_fname, pseudo_tt)

        self.clusters, self.noise = self.get_clusters(
            phases, pseudo_tt, max_search_dist, min_size
        )
        del pseudo_tt
        self.n_clusters = len(self.clusters)
        self.n_noise = len(self.noise)

    @staticmethod
    def compute_tt_matrix(phases, vmean):
        # optimization : matrix is symetric -> use lru_cache
        tt_matrix = []
        for p1 in tqdm(phases):
            line = []
            for p2 in phases:
                # line.append(compute_tt(p1, p2, vmean))
                line.append(compute_tt(*sorted((p1, p2)), vmean))
            tt_matrix.append(line)
        return tt_matrix

    @staticmethod
    def numpy_compute_tt_matrix(phases, vmean):
        # optimization : matrix is symetric -> use lru_cache
        nb_phases = len(phases)
        tt_matrix = np.empty([nb_phases, nb_phases], dtype=float)
        for i in range(0, nb_phases):
            p1 = phases[i]
            for j in range(0, nb_phases):
                p2 = phases[j]
                tt_matrix[i, j] = compute_tt(*sorted((p1, p2)), vmean)
        return tt_matrix

    @staticmethod
    def dask_compute_tt_matrix(phases, vmean):
        """Optimization to compute tt_matrix in //"""
        # data = [sorted((p1, p2)) for p1 in phases for p2 in phases]
        data = product(phases, repeat=2)
        b = db.from_sequence(data)
        tt_matrix_tmp = b.map(lambda x: compute_tt(*x, vmean)).compute()
        tt_matrix = np.array(tt_matrix_tmp).reshape((len(phases), len(phases)))
        return tt_matrix

    @staticmethod
    def get_clusters(phases, pseudo_tt, max_search_dist, min_size):
        # metric is “precomputed” ==> X is assumed to be a distance matrix and must be square
        db = DBSCAN(
            eps=max_search_dist, min_samples=min_size, metric="precomputed", n_jobs=-1
        ).fit(pseudo_tt)

        # db = OPTICS(
        #     min_cluster_size=6,
        #     eps=max_search_dist, min_samples=min_size, metric="precomputed", n_jobs=-1
        # ).fit(pseudo_tt)

        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        logger.info("Number of clusters: %d" % n_clusters_)
        logger.info("Number of noise points: %d" % n_noise_)

        cluster_ids = set(labels)

        # feed picks to associated clusters.
        # Take care of double picks ... keep only the one with the best probability
        clusters = []
        noise = []
        for c_id in cluster_ids:
            cluster = []
            for p, l in zip(phases, labels):
                if c_id == l:
                    cluster.append(p)

                    # It is better to rely on NonLinLoc to keep the relevant picks

                    # # check for duplicated station/phase pick
                    # # keep only the one with highest proba
                    # if len(cluster) == 0:
                    #     cluster.append(p)
                    #     continue

                    # to_remove = None
                    # to_insert = None

                    # for pp in cluster:
                    #     action = check_phase_take_over(p, pp)
                    #     if action == "takeover":
                    #         to_remove = pp
                    #         to_insert = p
                    #         break
                    #     elif action == "drop":
                    #         # do nothing
                    #         to_remove = None
                    #         to_insert = None
                    #         break
                    #     else:  # insert
                    #         to_remove = None
                    #         to_insert = p
                    #         # should wait until the end of the picks in cluster

                    # if to_insert:
                    #     cluster.append(p)
                    # if to_remove:
                    #     cluster.remove(pp)

            if c_id == -1:
                noise = cluster.copy()
            else:
                clusters.append(cluster)
        return clusters, noise

    def generate_nllobs(self, OBS_PATH):
        """
        export to obspy/NLL
        only 1 event/catalog (for NLL)
        """
        for i, cluster in enumerate(self.clusters):
            cat = Catalog()
            event = Event()
            stations_list = set([p.station for p in cluster])

            if self.min_station_count:
                if len(stations_list) < self.min_station_count:
                    logger.debug(
                        f"Cluster {i} ignored ... not enough stations ({len(stations_list)})"
                    )
                    continue

            for p in cluster:
                pick = Pick()
                pick.evaluation_mode = "automatic"
                pick.method_id = "phaseNet"
                pick.waveform_id = WaveformStreamID(
                    network_code=f"{p.network}", station_code=f"{p.station}"
                )
                pick.phase_hint = p.phase
                pick.time = p.time
                if "P" in p.phase and self.P_uncertainty:
                    pick.time_errors.uncertainty = self.P_uncertainty
                elif "S" in p.phase and self.S_uncertainty:
                    pick.time_errors.uncertainty = self.S_uncertainty
                event.picks.append(pick)

            cat.append(event)
            os.makedirs(OBS_PATH, exist_ok=True)
            obs_file = os.path.join(OBS_PATH, f"cluster-{i}.obs")
            logger.debug(f"Writting {obs_file} ({len(stations_list)})")
            cat.write(obs_file, format="NLLOC_OBS")

    def split_cluster(self):
        """split cluster for each duplicated phase"""
        pass

    def show_clusters(self):
        print(f"Clusters: {self.n_clusters}")
        for i, cluster in enumerate(self.clusters):
            stations_list = set([p.station for p in cluster])
            print(
                f"cluster {i}: %d picks / %d stations"
                % (len(self.clusters[i]), len(stations_list))
            )
            for p in sorted(cluster, key=lambda p: p.time):
                print(p)
            print("\n")

    def show_noise(self):
        print(f"Noise: {self.n_noise} picks")
        for i in self.noise:
            print(i)
        print("\n")


def _test():
    max_search_dist = 17.0
    min_size = 6
    average_velocity = 4.0  # km/s
    # phases = import_phases("../test/picks.csv")

    picks_file = "../test/EQT-2022-09-10.csv"
    logger.info(f"Opening {picks_file} file.")
    try:
        df = pd.read_csv(picks_file, parse_dates=["p_arrival_time", "s_arrival_time"])
    except Exception as e:
        logger.error(e)
        sys.exit()

    phases = import_eqt_phases(
        df,
        P_proba_threshold=0.8,
        S_proba_threshold=0.5,
    )
    logger.info(f"Read {len(phases)}")
    myclusters = Clusterize(phases, max_search_dist, min_size, average_velocity)
    myclusters.generate_nllobs("../test/obs")
    # myclusters.show_clusters()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    _test()
