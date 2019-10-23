"""
Calculate the best N for UMAP using local entropy
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.distance import euclidean

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

def calc_entropy(dists, bin_sz):
    maxd = np.quantile(dists, 0.99) + bin_sz
    bins = np.linspace(0,maxd, maxd/bin_sz)
    cts,b = np.histogram(dists, bins=bins)
    cts = cts.astype(np.float) / np.sum(cts).astype(np.float)
    entropy = -np.sum(cts * np.log(cts))
    return entropy

def get_pairwise_dists(nearest_dict, data, N):
    dists = []
    for k,v in nearest_dict.items():
        pts = [data[k]] + v[:N]
        for i,pt1 in enumerate(pts):
            for pt2 in pts[i+1:]:
                dists.append( euclidean(pt1,pt2) )
    return dists

def make_data(cfg):
    num_pts = cfg["num_points"]
    num_waves = cfg["num_waves"]
    x = np.linspace(0, 6*np.pi, num_pts)
    y = []
    for i in range(num_waves):
        y.append( np.sin(x) + i*0.25 )
    data = np.zeros((num_waves*num_pts,2))
    for i in range(num_waves):
        data[i*num_pts : (i+1)*num_pts, 0] = x
        data[i*num_pts : (i+1)*num_pts, 1] = y[i]
    noise = np.random.normal(0, cfg["sigma"], data.shape)
    data += noise
    return data

def make_nearest_dict(cfg, data):
    numr = cfg["num_starts"]
    N = len(data)
    idxs = np.random.choice(range(N), (numr,), replace=False)
    nearest = {}
    for idx in idxs:
        d = data[idx]
        metric_d = lambda d1 : euclidean(d,d1)
        pts = sorted(data, key=metric_d)[1:cfg["n_range"]+1]
        nearest[idx] = pts
    return nearest

def main(cfg):
    data = make_data(cfg)
    nearest_dict = make_nearest_dict(cfg, data)
    if cfg["test"]:
        plt.scatter(data[:,0], data[:,1], marker="o")
        for k,v in nearest_dict.items():
            plt.scatter(data[k][0], data[k][1], marker="+")
            near_x = [p[0] for p in v]
            near_y = [p[1] for p in v]
            plt.scatter(near_x, near_y, c="r", marker="s")
        plt.show()

    entropies = []
    for n in range(1,cfg["n_range"]+1):
        bin_sz = cfg["bin_size"]
        dists = get_pairwise_dists(nearest_dict, data, n)
        entropies.append( calc_entropy(dists, bin_sz) )
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
    parser.add_argument("--n-range", type=int, default=40)
    parser.add_argument("--bin-size", type=float, default=0.1)
    cfg = vars( parser.parse_args() )
    main(cfg)
