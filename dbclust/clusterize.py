#!/usr/bin/env python
import os
import sys
import logging
from math import pow, sqrt
import numpy as np
import pandas as pd
from itertools import product

# from sklearn.cluster import DBSCAN, OPTICS
import hdbscan
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
logger.setLevel(logging.INFO)


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


def filter_out_cluster_with_common_phases(clusters1, clusters2, min_com_phases):
    """
    Clusters are just a list of list(Phases).

    Phase must implement __eq__() to use set intersection.
    if cluster_stability is available use it, if not use cluster size
    to select the best cluster.
    """
    if clusters1.n_clusters == 0 or clusters2.n_clusters == 0:
        logger.debug("filter_out_cluster_with_common_phases: nothing to do ")
        return clusters1, clusters2, 0

    nb_cluster_removed = 0
    # for c1, c2 in product(clusters1.clusters, clusters2.clusters):
    for idx1, c1 in enumerate(clusters1.clusters):
        for idx2, c2 in enumerate(clusters2.clusters):
            intersection = set(c1).intersection(set(c2))
            logger.debug(f"Clusters c1, c2 share {len(intersection)} phases.")
            if len(intersection) >= min_com_phases:
                nb_cluster_removed += 1

                c1_stability = clusters1.clusters_stability[idx1]
                c2_stability = clusters2.clusters_stability[idx2]

                # if len(c1) > len(c2):
                if c1_stability > c2_stability:
                    cluster_removed = clusters2.clusters.pop(idx2)
                    cluster_removed_stability = c2_stability
                    clusters2.clusters_stability = np.delete(
                        clusters2.clusters_stability, idx2
                    )
                    #
                    cluster_kept = c1
                    cluster_kept_stability = clusters1.clusters_stability[idx1]
                else:
                    cluster_removed = clusters1.clusters.pop(idx1)
                    cluster_removed_stability = c1_stability
                    clusters1.clusters_stability = np.delete(
                        clusters1.clusters_stability, idx1
                    )
                    #
                    cluster_kept = c2
                    cluster_kept_stability = clusters2.clusters_stability[idx2]

                logger.info(
                    f"Keeping cluster with phases:{len(cluster_kept)}, stability:{cluster_kept_stability:.4f}, with first pick {cluster_kept[0]}"
                )
                logger.info(
                    f"Removing cluster with phases:{len(cluster_removed)}, stability:{cluster_removed_stability:.4f}, with first pick {cluster_removed[0]}"
                )
    return clusters1, clusters2, nb_cluster_removed


class Clusterize(object):
    def __init__(
        self,
        phases=None,
        min_cluster_size=5,  # hdbscan default
        average_velocity=5,  # km/s
        min_station_count=0,
        max_search_dist=0,  # same as hdbscan cluster_selection_epsilon: default is 0.
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
        self.clusters_stability = []
        self.n_clusters = 0
        self.noise = []
        self.n_noise = 0

        # clustering parameters
        self.max_search_dist = max_search_dist
        self.min_cluster_size = min_cluster_size
        self.average_velocity = average_velocity

        # pick filtering parameters
        self.min_station_count = min_station_count
        self.P_uncertainty = P_uncertainty
        self.S_uncertainty = S_uncertainty

        # tt_matrix load/save parameters
        self.tt_maxtrix_fname = tt_maxtrix_fname
        self.tt_matrix_load = tt_matrix_load
        self.tt_matrix_save = tt_matrix_save

        if phases is None:
            # Simple constructor
            return

        logger.info(
            f"Starting Clustering (nbphases={len(phases)}, min_cluster_size={min_cluster_size})."
        )
        if len(phases) < min_cluster_size:
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
            # pseudo_tt = self.compute_tt_matrix(phases, average_velocity)
            pseudo_tt = self.numpy_compute_tt_matrix(phases, average_velocity)
            # // computation using dask bag: slower for small cluster
            # pseudo_tt = self.dask_compute_tt_matrix(phases, average_velocity)
            logger.info(f"TT maxtrix: {compute_tt.cache_info()}")
            compute_tt.cache_clear()

        if tt_maxtrix_fname and tt_matrix_save:
            logger.info(f"Saving tt_matrix {tt_maxtrix_fname}.")
            np.save(tt_maxtrix_fname, pseudo_tt)

        self.clusters, self.clusters_stability, self.noise = self.get_clusters(
            phases, pseudo_tt, max_search_dist, min_cluster_size
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
    def get_clusters(phases, pseudo_tt, max_search_dist, min_cluster_size):
        # metric is ???precomputed??? ==> X is assumed to be a distance matrix and must be square

        # db = DBSCAN(
        #     eps=max_search_dist, min_samples=min_cluster_size, metric="precomputed", n_jobs=-1
        # ).fit(pseudo_tt)

        # db = OPTICS(
        #     min_samples=5,  # default value is 5 related to dbscan
        #     eps=max_search_dist,
        #     cluster_method="dbscan",
        #     algorithm='brute',
        #     metric="precomputed",
        #     n_jobs=-1,
        # ).fit(pseudo_tt)

        db = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,  # default 5
            # min_samples=None                          # default None
            allow_single_cluster=True,
            cluster_selection_epsilon=max_search_dist,  # default 0.0
            metric="precomputed",
            n_jobs=-1,
        ).fit(pseudo_tt)

        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        logger.info("Number of clusters: %d" % n_clusters_)
        logger.info("Number of noise points: %d" % n_noise_)

        cluster_ids = set(labels)

        # only for hdbscan
        # kind of cluster stability measurement [0, 1]
        if hasattr(db, "cluster_persistence_"):
            clusters_stability = db.cluster_persistence_
        else:
            clusters_stability = [1] * n_clusters_

        # feed picks to associated clusters.
        clusters = []
        noise = []
        for c_id in cluster_ids:
            cluster = []
            for p, l in zip(phases, labels):
                if c_id == l:
                    cluster.append(p)
                    # if duplicated picks, rely on NonLinLoc
                    # to keep the relevant picks at localisation level

            if c_id == -1:
                noise = cluster.copy()
            else:
                clusters.append(cluster)
        return clusters, clusters_stability, noise

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
                        f"Cluster {i}, stability:{self.clusters_stability[i]} ignored ... not enough stations ({len(stations_list)})"
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
            logger.debug(
                f"Cluster {i}, writting {obs_file}, stability:{self.clusters_stability[i]}, nstations:{len(stations_list)})"
            )
            cat.write(obs_file, format="NLLOC_OBS")

    def split_cluster(self):
        """split cluster for each duplicated phase"""
        pass

    def merge(self, clusters2):
        logger.info(
            f"Merging clusters list with {len(self.clusters)} and {len(clusters2.clusters)} clusters"
        )
        self.clusters += clusters2.clusters
        self.n_clusters = len(self.clusters)
        self.noise += clusters2.noise
        self.n_noise = len(self.noise)
        # clusters_stability are ndarray ... not a list
        self.clusters_stability = np.concatenate(
            (self.clusters_stability, clusters2.clusters_stability), axis=0
        )

    def show_clusters(self):
        print(f"Clusters: {self.n_clusters}")
        for i, cluster in enumerate(self.clusters):
            stations_list = set([p.station for p in cluster])
            print(
                f"cluster {i}: stability=%.2f, %d picks / %d stations"
                % (
                    self.clusters_stability[i],
                    len(self.clusters[i]),
                    len(stations_list),
                )
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
    min_cluster_size = 6
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
    # myclusters = Clusterize(phases, min_cluster_size, average_velocity, max_search_dist)
    myclusters = Clusterize(
        phases=phases,
        # max_search_dist=max_search_dist,
        max_search_dist=0,
        # min_cluster_size=min_cluster_size,
        min_cluster_size=5,
        average_velocity=average_velocity,
    )
    myclusters.generate_nllobs("../test/obs")
    myclusters.show_clusters()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    _test()
