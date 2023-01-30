#!/usr/bin/env python
import os
import sys
import logging
from math import pow, sqrt
import numpy as np
import pandas as pd
#from numba import jit
from itertools import product
from sklearn.cluster import DBSCAN, OPTICS
from tqdm import tqdm
import functools
import dask.bag as db
from dask.cache import Cache
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


class Clusterize(object):
    def __init__(
        self,
        phases,
        max_search_dist,
        min_size,
        average_velocity,
        tt_maxtrix_fname="tt_matrix.npy",
        tt_matrix_load=False,
        tt_matrix_save=False,
    ):
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
            # pseudo_tt = self.numpy_compute_tt_matrix(phases, average_velocity)
            # // computation using dask bag
            pseudo_tt = self.dask_compute_tt_matrix(phases, average_velocity)

        if tt_maxtrix_fname and tt_matrix_save:
            logger.info(f"Saving tt_matrix {tt_maxtrix_fname}.")
            np.save(tt_maxtrix_fname, pseudo_tt)
        logger.info(compute_tt.cache_info())

        logger.info("Starting Clustering.")
        self.clusters, self.noise = self.get_clusters(
            phases, pseudo_tt, max_search_dist, min_size
        )
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
    #@jit(nopython=True) 
    def numpy_compute_tt_matrix(phases, vmean):
        # optimization : matrix is symetric -> use lru_cache
        tt_matrix = np.empty([len(phases), len(phases)], dtype=float)
        for i, p1 in tqdm(enumerate(phases)):
            for j, p2 in enumerate(phases):
                tt_matrix[i,j] = compute_tt(*sorted((p1, p2)), vmean)
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

        clusters = []
        for c_id in cluster_ids:
            cluster = []
            for p, l in zip(phases, labels):
                if c_id == l:
                    cluster.append(p)
            if c_id == -1:
                noise = cluster.copy()
            else:
                clusters.append(cluster)
        return clusters, noise

    def generate_nllobs(
        self, OBS_PATH, min_station_count, P_uncertainty=None, S_uncertainty=None
    ):
        """
        export to obspy/NLL
        only 1 event/catalog (for NLL)
        """
        for i, cluster in tqdm(enumerate(self.clusters)):
            cat = Catalog()
            event = Event()
            stations_list = set([p.station for p in cluster])
            if len(stations_list) < min_station_count:
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
                if "P" in p.phase and P_uncertainty:
                    pick.time_errors.uncertainty = P_uncertainty
                elif "S" in p.phase and S_uncertainty:
                    pick.time_errors.uncertainty = S_uncertainty
                event.picks.append(pick)
            cat.append(event)
            os.makedirs(OBS_PATH, exist_ok=True)
            obs_file = os.path.join(OBS_PATH, f"cluster-{i}.obs")
            logger.debug(f"Writting {obs_file}")
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
        proba_threshold=0.8,
    )
    logger.info(f"Read {len(phases)}")
    Clusterize(phases, max_search_dist, min_size, average_velocity)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    _test()
