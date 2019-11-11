"""
Show the distribution of different properties of a distribution as a function of
n nearest neighbors
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import OrderedDict
from scipy.spatial.distance import euclidean

sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
import wg_utils as U

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def plot_dists(prop_dict, num_bins):
    N = len(prop_dict)
    figsize = (10,8)
    num_rows = int( np.floor( np.sqrt(N) ) )
    num_cols = int( np.ceil(N/num_rows) )
    fig,axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    prop_list = list( prop_dict.items() )
    for ct in range(N):
        i = ct // num_cols
        j = ct % num_cols
        ax = axs[i][j]
        n = prop_list[ct][0]
        dist = prop_list[ct][1]
        bins = np.linspace(0, np.quantile(dist, 0.99), num_bins)
        ax.set_title("n = %d" % n)
        ax.hist(dist, bins=bins)

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])


def main(cfg):
    data = U.make_wave_data(cfg["num_waves"], cfg["num_points"], cfg["sigma"],
            cfg["offset"])
    nearest_dict = U.make_nearest_dict(cfg["num_starts"], cfg["n_range"], data)
    prop_dict = OrderedDict()
    for n in range(1,cfg["n_range"]+1):
        if cfg["property"] == "pwise-dist":
            dists = U.get_pairwise_dists(nearest_dict, data, n)
        elif cfg["property"] == "dist0":
            dists = U.get_dists(nearest_dict, data, n)
        else:
            raise NotImplementedError( cfg["property"] )
        prop_dict[n] = dists        
        
    plot_dists(prop_dict, cfg["num_bins"])

    plt.suptitle("Distribution '%s' as function of n" % cfg["property"])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--num-waves", type=int, default=4)
    parser.add_argument("--num-starts", type=int, default=10)
    parser.add_argument("--offset", type=float, default=0.25)
    parser.add_argument("--n-range", type=int, default=40)
    parser.add_argument("--num-bins", type=int, default=50)

    parser.add_argument("--property", default="pwise-dist",
            choices=["pwise-dist", "dist0"])

    cfg = vars( parser.parse_args() )
    main(cfg)

