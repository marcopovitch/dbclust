#!/usr/bin/env python
import os
import sys
import logging
from math import pow, sqrt, isnan
import numpy as np
import pandas as pd
from itertools import product, chain

# from sklearn.cluster import DBSCAN, OPTICS
import hdbscan
from tqdm import tqdm
import functools
from itertools import combinations
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


def cluster_share_eventid(c1, c2):
    c1_evtids = set([p.eventid for p in c1 if p.eventid])
    logger.debug("cluster1 eventid: %s" % c1_evtids)
    c2_evtids = set([p.eventid for p in c2 if p.eventid])
    logger.debug("cluster2 eventid: %s" % c2_evtids)
    if c1_evtids.intersection(c2_evtids):
        return True
    else:
        return False


def manage_cluster_with_common_phases(clusters1, clusters2, min_com_phases):
    """
    Clusters are just a list of list(Phases).

    Phase must implement __eq__() to use set intersection.
    """
    if clusters1.n_clusters == 0 or clusters2.n_clusters == 0:
        logger.debug("filter_out_cluster_with_common_phases: nothing to do ")
        return clusters1, clusters2, 0

    logger.info(f"Clusters c1 contain {len(clusters1.clusters)} clusters")
    logger.info(f"Clusters c2 contain {len(clusters2.clusters)} clusters")

    c1_used = set()
    c2_used = set()

    merge_count = 0
    for idx1, c1 in enumerate(clusters1.clusters):
        if idx1 in c1_used:
            continue
        for idx2, c2 in enumerate(clusters2.clusters):
            if idx2 in c2_used or idx1 in c1_used:
                continue

            logger.debug("Intersection c1, c2:")
            logger.debug("c1[%d] = %s" % (idx1, c1))
            logger.debug("c2[%d] = %s" % (idx2, c2))

            nb_intersection = set(c1).intersection(set(c2))
            eventid_shared = cluster_share_eventid(c1, c2)

            if not eventid_shared and len(nb_intersection) < min_com_phases:
                logger.debug(
                    f"Clusters c1[{idx1}], c2[{idx2}] have only {len(nb_intersection)} shared phases (merge require {min_com_phases})."
                    f"They share no picks with an EventId in common."
                )
                continue

            logger.info(
                f"Merging clusters c1[{idx1}], c2[{idx2}]: "
                f"{len(nb_intersection)} shared phases, picks with an EventId in common is {eventid_shared}"
            )

            merge_count += 1
            c1_used.add(idx1)
            c2_used.add(idx2)

            # merge 2 clusters and remove duplicated picks
            set_c1 = set(clusters1.clusters[idx1])
            set_c1.update(set(clusters2.clusters[idx2]))
            clusters1.clusters[idx1] = list(set_c1)
            clusters1.n_clusters = len(clusters1.clusters)

            # keep cluster stability
            c1_stab = clusters1.clusters_stability[idx1]
            c2_stab = clusters2.clusters_stability[idx2]

            # keep the best stability for c1
            try:
                clusters1.clusters_stability[idx1] = max(c1_stab, c2_stab)
            except Exception as e:
                logger.error(e)
                print("c1_stab", c1_stab)
                print("c1_stab", c2_stab)
                sys.exit(255)

            logger.info(
                f"[{idx1},{idx2}] merging cluster(len:{len(c1)}, stability:{c1_stab:.4f}) "
                f"with cluster(len:{len(c2)}, stability:{c2_stab:.4f}) "
                f"--> cluster(len:{len(clusters1.clusters[idx1])}, stability:{clusters1.clusters_stability[idx1]:.4f})"
            )

    # Do the real cleanup of c2
    for idx2 in sorted(c2_used, reverse=True):
        clusters2.clusters.pop(idx2)
        clusters2.clusters_stability = np.delete(clusters2.clusters_stability, idx2)
    clusters2.n_clusters = len(clusters2.clusters) 

    return clusters1, clusters2, merge_count


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
            logger.info(f"Clusters c1, c2 share {len(intersection)} phases.")
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
        min_station_with_P_and_S=2,
        max_search_dist=0,  # same as hdbscan cluster_selection_epsilon: default is 0.
        P_uncertainty=0.1,
        S_uncertainty=0.2,
        tt_maxtrix_fname="tt_matrix.npy",
        tt_matrix_load=False,
        tt_matrix_save=False,
        log_level=logging.DEBUG,
    ):
        logger.setLevel(log_level)

        # clusters is a list of cluster :
        # ie. [ [phases, label], ... ]
        # noise is [ phases, -1]
        self.clusters = []
        self.clusters_stability = np.array([])
        self.n_clusters = 0
        self.noise = []
        self.n_noise = 0

        # clustering parameters
        self.max_search_dist = max_search_dist
        self.min_cluster_size = min_cluster_size
        self.average_velocity = average_velocity

        # stations filtering parameters
        self.min_station_count = min_station_count
        self.min_station_with_P_and_S = min_station_with_P_and_S

        # pick filtering parameters
        self.P_uncertainty = P_uncertainty
        self.S_uncertainty = S_uncertainty

        # tt_matrix load/save parameters
        self.tt_maxtrix_fname = tt_maxtrix_fname
        self.tt_matrix_load = tt_matrix_load

        if phases is None:
            # Simple constructor
            return

        logger.info(
            f"Starting Clustering (nbphases={len(phases)}, "
            f"min_cluster_size={min_cluster_size}, "
            f"min_station_with_P_and_S={min_station_with_P_and_S})."
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
        self.n_clusters = len(self.clusters)
        self.n_noise = len(self.noise)
        del pseudo_tt
        self.cluster_merge_based_on_eventid()


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
        # metric is “precomputed” ==> X is assumed to be a distance matrix and must be square

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
                    # or use the pick probability

            if c_id == -1:
                noise = cluster.copy()
            else:
                clusters.append(cluster)
                logger.debug(f"cluster[{c_id}]: {len(cluster)} phases")

        return clusters, clusters_stability, noise

    def cluster_merge_based_on_eventid(self):
        """Merge clusters containing picks from the same eventid (if provided)."""

        # logger.error(f"n_clusters={self.n_clusters}, lens(clusters) = {len(self.clusters)}") 
        # assert self.n_clusters == len(self.clusters)

        if self.n_clusters <= 1:
            return

        logger.info("Merging clusters sharing same EventId.")
        logger.info(f"Working on {self.n_clusters} clusters.")

        final_cluster_list = []
        while self.clusters:
            clusters_to_merge = []
            c1 = self.clusters.pop(0)
            clusters_to_merge.append(c1)
            cluster_to_remove = []
            logger.debug("Working on cluster %s with %d phases" % (c1, len(c1)))
            for i, c2 in enumerate(self.clusters):
                if cluster_share_eventid(c1, c2):
                    # logger.debug("Eventid shared.")
                    cluster_to_remove.append(c2)
                    clusters_to_merge.append(c2)
                # else:
                # logger.debug("No eventid shared.")

            # cleanup
            for c in cluster_to_remove:
                self.clusters.remove(c)

            # merge clusters
            new_cluster = list(chain(*clusters_to_merge))
            final_cluster_list.append(new_cluster)

        # Sanity check: self.clusters should be empty
        assert not len(self.clusters)
        self.clusters = final_cluster_list
        self.n_clusters = len(self.clusters)
        self.clusters_stability = np.full(self.n_clusters, 1.0)
        logger.info(f"EventId merge leads to {self.n_clusters} clusters.")

    def generate_nllobs(self, OBS_PATH):
        """
        export to obspy/NLL
        only 1 event/catalog (for NLL)
        """
        for i, cluster in enumerate(self.clusters):
            cat = Catalog()
            event = Event()

            # count the number of stations
            stations_list = set([p.station for p in cluster])
            if self.min_station_count:
                if len(stations_list) < self.min_station_count:
                    logger.debug(
                        f"Cluster {i}, stability:{self.clusters_stability[i]} ignored ... "
                        f"not enough stations ({len(stations_list)}/{self.min_station_count})"
                    )
                    continue

            # count the number of station that have both P and S
            if self.min_station_with_P_and_S:
                stations_with_P_and_S_count = 0
                for s in stations_list:
                    phase_list = set([p.phase for p in cluster if p.station == s])
                    # Pn, Pg count as P,
                    # Sn, Sg count as S
                    phase_list = set(
                        ["P" for i in phase_list if "P" in i]
                        + ["S" for i in phase_list if "S" in i]
                    )
                    if len(phase_list) == 2:
                        stations_with_P_and_S_count += 1
                if stations_with_P_and_S_count < self.min_station_with_P_and_S:
                    logger.debug(
                        f"Cluster {i}, stability:{self.clusters_stability[i]} ignored ... "
                        f"not enough stations with both P and S ({stations_with_P_and_S_count}/{self.min_station_with_P_and_S})"
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
            f"Merging clusters list: {len(self.clusters)} clusters from list1 and {len(clusters2.clusters)} from list2"
        )

        self.clusters += clusters2.clusters
        self.n_clusters = len(self.clusters)
        self.noise += clusters2.noise
        self.n_noise = len(self.noise)

        # print(self.clusters_stability)
        # print(clusters2.clusters_stability)

        # clusters_stability are ndarray ... not a list : should be fixed !
        self.clusters_stability = np.array(
            self.clusters_stability.tolist() + clusters2.clusters_stability.tolist()
        )
        # self.show_clusters()

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
    myclusters = Clusterize(
        phases=phases,
        # max_search_dist=max_search_dist,
        max_search_dist=0,
        # min_cluster_size=min_cluster_size,
        # min_station_with_P_and_S=2,
        min_cluster_size=5,
        average_velocity=average_velocity,
    )
    myclusters.generate_nllobs("../test/obs")
    myclusters.show_clusters()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    _test()
