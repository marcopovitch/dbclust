#!/usr/bin/env python
import functools
import json
import logging
import os
import sys
from collections import Counter
from itertools import chain
from itertools import product
from math import isnan
from math import pow
from math import sqrt
from typing import List
from typing import Optional
from typing import Union

import dask.bag as db
import hdbscan
import numpy as np
import pandas as pd
import ray
from icecream import ic
from obspy import Catalog
from obspy.core.event import Comment
from obspy.core.event import CreationInfo
from obspy.core.event import Event
from obspy.core.event import Origin
from obspy.core.event.base import WaveformStreamID
from obspy.core.event.origin import Pick
from obspy.geodetics import gps2dist_azimuth
from phase import import_eqt_phases
from phase import import_phases
from phase import Phase
from tqdm import tqdm

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("clusterize")
logger.setLevel(logging.INFO)


@functools.lru_cache(maxsize=None)
def compute_tt(p1: Phase, p2: Phase, vmean) -> float:
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


def cluster_share_eventid(c1: List[Phase], c2: List[Phase]) -> bool:
    c1_evtids = set([p.event_id for p in c1 if p.event_id])
    logger.debug("cluster1 eventid: %s" % c1_evtids)
    c2_evtids = set([p.event_id for p in c2 if p.event_id])
    logger.debug("cluster2 eventid: %s" % c2_evtids)
    if c1_evtids.intersection(c2_evtids):
        return True
    else:
        return False


def get_picks_from_event(event: Event, origin: Origin, time) -> List:
    # station_id,phase_type,phase_time
    # 1K.OFAS0.00.EH.D,P,2023-02-13T18:30:58.558999Z
    lines = []
    for arrival in origin.arrivals:
        if arrival.time_weight and arrival.time_residual:
            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            if pick:
                if time and pick.time >= time:
                    line = [
                        pick.waveform_id.get_seed_string(),
                        pick.phase_hint,
                        pick.time,
                    ]
                else:
                    line = [
                        pick.waveform_id.get_seed_string(),
                        pick.phase_hint,
                        pick.time,
                    ]
                lines.append(line)
    return sorted(lines, key=lambda l: l[2])


def feed_picks_probabilities(cat: Catalog, clusters: List[List[Phase]]) -> None:
    for event in cat:
        for pick in event.picks:
            for p in set(chain(*clusters)):
                # pick_seedid = ".".join(
                #     [pick.waveform_id["network_code"], pick.waveform_id["station_code"]]
                # )
                # p_seedid = ".".join([p.network, p.station])
                # if (
                #     pick_seedid == p_seedid
                #     and pick.time == p.time
                #     and pick.phase_hint == p.phase
                # ):
                if (
                    pick.waveform_id["station_code"] == p.station
                    and pick.time == p.time
                    and pick.phase_hint == p.phase
                ):
                    if pick.waveform_id["network_code"] != p.network:
                        logger.warning(
                            f"Check your inventory for station {p.station}, 2 networks defined : "
                            f"[{pick.waveform_id['network_code']},{p.network}] "
                        )
                    if p.agency:
                        agency = p.agency
                    else:
                        agency = "undefined"
                    pick.comments.append(
                        Comment(
                            text='{"probability": {"name": "%s", "value": %.2f}}'
                            % (agency, p.proba)
                        )
                    )


def feed_picks_event_ids(cat: Catalog, clusters: List[List[Phase]]) -> None:
    for event in cat:
        o = event.preferred_origin()
        cluster_found = False
        event_ids = []
        for a in o.arrivals:
            if cluster_found == True:
                break
            if a.time_weight and a.time_residual:
                pick = next(
                    (p for p in event.picks if p.resource_id == a.pick_id), None
                )
                if pick is None:
                    continue
                for c in clusters:
                    for cluster_pick in c:
                        if (
                            pick.time == cluster_pick.time
                            and pick.phase_hint == cluster_pick.phase
                        ):
                            # cluster found
                            event_ids = list(set([p.event_id for p in c if p.event_id]))
                            cluster_found = True
                            break
                    if cluster_found == True:
                        break

        # event_ids = list(set([p.event_id for p in chain(*clusters) if p.event_id]))
        event.comments.append(Comment(text='{"event_ids": %s}' % json.dumps(event_ids)))


def merge_cluster_with_common_phases(
    clusters1: List[Phase], clusters2: List[Phase], min_com_phases: int
) -> (List[Phase], List[Phase], int):
    """
    Merge into clusters1 all clusters from clusters2 with common phases
    or shared event id.

    Clusters are just a list of list(Phases).
    Phase must implement __eq__() to use set intersection.

    Returns merged clusters1, remaining unmerged clusters in clusters2,
    and the number of merge realized.
    """
    new_cluster2 = []
    merge_count = 0
    logger.debug(
        "merge_cluster_with_common_phases: clusters1 contains %d clusters."
        % len(clusters1.clusters)
    )
    logger.debug(
        "merge_cluster_with_common_phases: clusters2 contains %d clusters."
        % len(clusters2.clusters)
    )

    for idx2, c2 in enumerate(clusters2.clusters):
        merged_flag = False
        for idx1, c1 in enumerate(clusters1.clusters):
            common_elements = (Counter(c1) & Counter(c2)).values()
            common_count = sum(common_elements)

            eventid_shared = cluster_share_eventid(c1, c2)
            if common_count >= min_com_phases or eventid_shared:
                logger.debug(
                    f"merging c2[{idx2}] (myclust) into c1[{idx1}] (previous_myclust): "
                    f"picks shared: {common_count}, eventid shares: {eventid_shared}"
                )
                c1.extend(c2)
                c1 = list(set(c1))
                merge_count += 1
                merged_flag = True
                break
        if not merged_flag:
            new_cluster2.append(c2)

    clusters2.clusters = new_cluster2
    clusters2.n_clusters = len(clusters2.clusters)
    # generate "articifial" cluster stability information (but not used !)
    clusters2.clusters_stability = np.full(clusters2.n_clusters, 1.0)
    clusters1.clusters_stability = np.full(clusters1.n_clusters, 1.0)

    logger.debug("merge_cluster_with_common_phases: merge_count %d" % merge_count)

    return clusters1, clusters2, merge_count


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
        tt_matrix_fname="tt_matrix.npy",
        tt_matrix_load=False,
        tt_matrix_save=False,
        zones=None,
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
        self.preloc = None  # pre-localization if pyocto was enable
        self.zones = zones

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
        self.tt_matrix_fname = tt_matrix_fname
        self.tt_matrix_load = tt_matrix_load

        if phases is None:
            # Simple constructor
            return

        logger.info(
            f"Starting Clustering (nb phases={len(phases)}, "
            f"min_cluster_size={min_cluster_size}, "
            f"min_station_with_P_and_S={min_station_with_P_and_S})."
        )
        if len(phases) < min_cluster_size:
            logger.info(f"Too few picks ({len(phases)}/{min_cluster_size})!")
            # add noise points
            self.clusters = []
            self.n_clusters = 0
            self.noise = [phases, -1]
            self.n_noise = len(phases)
            return

        logger.info("Computing TT matrix.")
        if tt_matrix_load and tt_matrix_fname:
            logger.info(f"Loading tt_matrix {tt_matrix_fname}.")
            try:
                pseudo_tt = np.load(tt_matrix_fname)
            except Exception as e:
                logger.error(e)
                logger.error("Check your config file !")
                sys.exit()
        else:
            # sequential computation
            # don't forget to activate lru_cache for compute_tt()
            # pseudo_tt = self.compute_tt_matrix(phases, average_velocity)
            # pseudo_tt = self.numpy_compute_tt_matrix_seq(phases, average_velocity)

            # use the fact that the matrix is diagonal and symmetrical
            # running time is quite similar to sequential computation + lru_cache
            pseudo_tt = self.numpy_compute_tt_matrix(phases, average_velocity)

            # // computation using dask bag: slower for small cluster
            # pseudo_tt = self.dask_compute_tt_matrix(phases, average_velocity)
            try:
                logger.info(f"TT matrix: {compute_tt.cache_info()}")
                compute_tt.cache_clear()
            except:
                pass

        if tt_matrix_fname and tt_matrix_save:
            logger.info(f"Saving tt_matrix {tt_matrix_fname}.")
            np.save(tt_matrix_fname, pseudo_tt)

        self.clusters, self.clusters_stability, self.noise = self.get_clusters(
            phases, pseudo_tt, max_search_dist, min_cluster_size
        )
        self.n_clusters = len(self.clusters)
        self.n_noise = len(self.noise)
        del pseudo_tt
        self.cluster_merge_based_on_eventid()

    @staticmethod
    def compute_tt_matrix(phases, vmean):
        # optimization : matrix is symmetrical -> use lru_cache
        tt_matrix = []
        for p1 in tqdm(phases):
            line = []
            for p2 in phases:
                # line.append(compute_tt(p1, p2, vmean))
                line.append(compute_tt(*sorted((p1, p2)), vmean))
            tt_matrix.append(line)
        return tt_matrix

    @staticmethod
    def numpy_compute_tt_matrix_seq(phases, vmean):
        # optimization : matrix is symmetrical -> use lru_cache
        nb_phases = len(phases)
        tt_matrix = np.empty([nb_phases, nb_phases], dtype=float)
        for i in range(0, nb_phases):
            p1 = phases[i]
            for j in range(0, nb_phases):
                p2 = phases[j]
                tt_matrix[i, j] = compute_tt(*sorted((p1, p2)), vmean)
        return tt_matrix

    @staticmethod
    def numpy_compute_tt_matrix(phases, vmean):
        # optimization : matrix is diagonal and symmetrical
        nb_phases = len(phases)
        elements = []
        for i in range(0, nb_phases):
            p1 = phases[i]
            for j in range(i, nb_phases):
                p2 = phases[j]
                elements.append(compute_tt(*sorted((p1, p2)), vmean))

        matrix_upper1 = np.zeros((nb_phases, nb_phases))
        row, col = np.triu_indices(nb_phases)
        matrix_upper1[row, col] = elements

        matrix_upper2 = np.copy(matrix_upper1)
        np.fill_diagonal(matrix_upper2, 0)
        tt_matrix = matrix_upper1 + matrix_upper2.T
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
                    # to keep the relevant picks at localization level
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

        logger.info(
            "cluster_merge_based_on_eventid(): merging clusters (myclust) sharing same EventId."
        )
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
                pick.creation_info = CreationInfo(agencyID=p.agency)
                if p.evaluation in ["automatic", "manual"]:
                    pick.evaluation_mode = p.evaluation
                pick.method_id = p.method
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
                f"Cluster {i}, writing {obs_file}, stability:{self.clusters_stability[i]}, n_stations:{len(stations_list)})"
            )
            cat.write(obs_file, format="NLLOC_OBS")

            # use pyocto pre-localization to select velocity model to be used
            # create vel_file with required information
            if self.preloc:
                hypo = self.preloc[i]
                logger.info(
                    f"Prelocalization is time={hypo['time']}, lat={hypo['latitude']}, "
                    f"lon={hypo['longitude']}, depth_m={hypo['depth_m']}"
                )
                if not self.zones.polygons.empty:
                    zone, min_dist_km = self.zones.find_zone(
                        latitude=hypo["latitude"],
                        longitude=hypo["longitude"],
                    )

                    # If preloc is too close from an polygon edge
                    # use the national velocity model
                    # if min_dist_km < 100:
                    #     logger.info(f"Preloc is close ({min_dist_km} km)"
                    #                 f" to '{zone['name']}' polygon edge ! Using 'world' zone")
                    #     zone = self.zones.get_zone_from_name("world")
                    #     min_dist_km = None

                    if not zone.empty:
                        logger.info(
                            f"Using zone:\n"
                            f"\tname: '{zone['name']}'\n"
                            f"\twith velocity profile: '{zone['velocity_profile']}'\n"
                            f"\ttemplate: '{zone['template']}'\n"
                            f"\tmin_dist_km: {min_dist_km}"
                        )

                        vel_file = os.path.join(OBS_PATH, f"cluster-{i}.vel")
                        logger.debug(f"writing to file {vel_file}: {zone['template']}")
                        with open(vel_file, "w") as vel:
                            vel.write(zone["velocity_profile"] + "\n")
                            vel.write(zone["template"] + "\n")
                            vel.write(f"{hypo['time']}\n")
                            vel.write(f"{hypo['latitude']}\n")
                            vel.write(f"{hypo['longitude']}\n")
                            vel.write(f"{hypo['depth_m']}\n")
                    else:
                        logger.warning(
                            f"Can't find a matched zone for (lat={hypo['latitude']},lon={hypo['longitude']})!"
                        )

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
        print(f"Clusters: number of clusters = {self.n_clusters}")
        for i, cluster in enumerate(self.clusters):
            stations_list = set([p.station for p in cluster])
            evtids = set([p.event_id for p in cluster if p.event_id])
            print(
                f"\tcluster {i}: stability=%.2f, %d picks / %d stations, eventids: %s"
                % (
                    self.clusters_stability[i],
                    len(self.clusters[i]),
                    len(stations_list),
                    evtids,
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
    average_velocity = 5.0  # km/s

    picks_file = "../samples/renass.csv"
    logger.info(f"Opening {picks_file} file.")
    try:
        df = pd.read_csv(picks_file, parse_dates=["phase_time"])
    except Exception as e:
        logger.error(e)
        sys.exit()

    logger.info(f"Read {len(df)} phases.")
    phases = import_phases(
        df,
        P_proba_threshold=0.3,
        S_proba_threshold=0.3,
        info_sta="http://10.0.1.36:8080",
        # info_sta="http://ws.resif.fr",
    )

    myclusters = Clusterize(
        phases=phases,
        max_search_dist=60,
        min_station_with_P_and_S=1,
        min_cluster_size=3,
        average_velocity=average_velocity,
    )
    myclusters.generate_nllobs("../test/obs")
    myclusters.show_clusters()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    _test()
