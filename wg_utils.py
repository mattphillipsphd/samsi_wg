"""
Utilities
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

def make_nearest_dict(num_starts, n_range, data):
    numr = num_starts
    N = len(data)
    idxs = np.random.choice(range(N), (numr,), replace=False)
    nearest = {}
    for idx in idxs:
        d = data[idx]
        metric_d = lambda d1 : euclidean(d,d1)
        pts = sorted(data, key=metric_d)[1:n_range+1]
        nearest[idx] = pts
    return nearest

def make_wave_data(num_waves, num_pts, noise_sigma):
    x = np.linspace(0, 6*np.pi, num_pts)
    y = []
    for i in range(num_waves):
        y.append( np.sin(x) + i*0.25 )
    data = np.zeros((num_waves*num_pts,2))
    for i in range(num_waves):
        data[i*num_pts : (i+1)*num_pts, 0] = x
        data[i*num_pts : (i+1)*num_pts, 1] = y[i]
    noise = np.random.normal(0, noise_sigma, data.shape)
    data += noise
    return data


