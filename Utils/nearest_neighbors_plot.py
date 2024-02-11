#!/usr/bin/env python
import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.neighbors import NearestNeighbors


def ymljoin(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


def yml_read_config(filename):
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("maxsd_plot")
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        dest="configfile",
        help="yaml configuration file.",
        type=str,
    )
    args = parser.parse_args()
    if not args.configfile:
        parser.print_help()
        sys.exit(255)

    yaml.add_constructor("!join", ymljoin)
    cfg = yml_read_config(args.configfile)
    cluster_cfg = cfg["cluster"]

    min_cluster_size = cluster_cfg["min_cluster_size"]
    if "pre_computed_tt_matrix" in cluster_cfg:
        pre_computed_tt_matrix = cluster_cfg["pre_computed_tt_matrix"]
    else:
        logger.error(f"No tt_matrix defined in configuration file")
        sys.exit(255)

    if "max_search_dist" in cluster_cfg:
        max_search_dist = cluster_cfg["max_search_dist"]
    else:
        max_search_dist = 0.0  # default for hdbscan

    tt_matrix = np.load(pre_computed_tt_matrix)
    min_size = min_cluster_size

    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    # n_neighbors :number of neighbors to use by default
    nbrs = NearestNeighbors(n_neighbors=min_size, metric="precomputed", n_jobs=-1).fit(
        tt_matrix
    )

    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(tt_matrix)

    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)

    k_dist = sort_neigh_dist[:, min_size - 1]
    plt.plot(k_dist)

    plt.axhline(y=max_search_dist, linewidth=1, linestyle="dashed", color="k", label='max-search-dist')
    plt.title(f"k-NearestNeighbors distance using min_size={min_size}")
    plt.ylabel("k-NN distance")
    plt.xlabel(f"Sorted observations (ie. P & S phases)")
    plt.legend()
    plt.show()
