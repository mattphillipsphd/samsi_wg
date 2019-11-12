"""
Calculate the best N for UMAP using local entropy
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.spatial.distance import euclidean

sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
import wg_utils as U

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

def main(cfg):
    data = U.make_wave_data(cfg["num_waves"], cfg["num_points"], cfg["sigma"],
            cfg["offset"])
    nearest_dict = U.make_nearest_dict(cfg["num_starts"], cfg["n_range"], data)
    if cfg["test"]:
        num_pts = cfg["num_points"]
        colors = ["r", "b", "g", "m", "c", "k", "y"]
        for w in range(cfg["num_waves"]):
            plt.scatter(data[ w*num_pts : (w+1)*num_pts, 0 ],
                    data[ w*num_pts : (w+1)*num_pts :, 1 ], marker="o",
                    color=colors[w%len(colors)])
        for k,v in nearest_dict.items():
            plt.scatter(data[k][0], data[k][1], marker="+")
            near_x = [p[0] for p in v]
            near_y = [p[1] for p in v]
            plt.scatter(near_x, near_y, c="r", marker="s")
        plt.show()

    entropies = []
    for n in range(1,cfg["n_range"]+1):
        bin_sz = cfg["bin_size"]
        dists = U.get_pairwise_dists(nearest_dict, data, n)
        entropies.append( U.calc_entropy(dists, bin_sz) )
    plt.plot(list(range(1,cfg["n_range"]+1)), entropies)
    plt.title("Local entropy as function of N nearest neighbors")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--num-waves", type=int, default=4)
    parser.add_argument("--num-starts", type=int, default=10)
    parser.add_argument("--offset", type=float, default=0.5)
    parser.add_argument("--n-range", type=int, default=40)
    parser.add_argument("--bin-size", type=float, default=0.1)
    cfg = vars( parser.parse_args() )
    main(cfg)
