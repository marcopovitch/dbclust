#!/usr/bin/env python
import os
import sys
import logging
from math import pow, sqrt
from sklearn.cluster import DBSCAN

from obspy import Catalog
from obspy.core.event import Event
from obspy.core.event.base import WaveformStreamID
from obspy.core.event.origin import Pick
from obspy.geodetics import gps2dist_azimuth

from dbclust.phase import import_phases

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("clusterize")
logger.setLevel(logging.DEBUG)


class Clusterize(object):
    def __init__(self, phases, max_search_dist, min_size, average_velocity):
        logger.debug("Computing TT matrix")
        pseudo_tt = self.compute_tt_matrix(phases, average_velocity)
        logger.debug("Clustering ...")
        self.clusters, self.noise = self.get_clusters(
            phases, pseudo_tt, max_search_dist, min_size
        )
        self.n_clusters = len(self.clusters)
        self.n_noise = len(self.noise)

    @staticmethod
    def compute_tt_matrix(phases, vmean):
        tt_matrix = []
        for p1 in phases:
            line = []
            for p2 in phases:
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
                line.append(tt)
            tt_matrix.append(line)
        return tt_matrix

    @staticmethod
    def get_clusters(phases, pseudo_tt, max_search_dist, min_size):
        # metric is “precomputed” ==> X is assumed to be a distance matrix and must be square
        db = DBSCAN(
            eps=max_search_dist, min_samples=min_size, metric="precomputed"
        ).fit(pseudo_tt)
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
        for i, cluster in enumerate(self.clusters):
            cat = Catalog()
            event = Event()
            stations_list = set([p.station for p in cluster])
            if len(stations_list) < min_station_count:
                logger.info(
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
                p.oneline_show()
            print("\n")

    def show_noise(self):
        print(f"Noise: {self.n_noise} picks")
        for i in self.noise:
            i.oneline_show()
        print("\n")


def _test():
    max_search_dist = 17.0
    min_size = 6
    average_velocity = 4.0  # km/s
    phases = import_phases("../test/picks.csv")
    Clusterize(phases, max_search_dist, min_size, average_velocity)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    _test()
