"""
Utilities
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import OrderedDict
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

# Inputs
#   pts: (N, D) numpy array, where N is the number of points and D is the number
#       of dimensions
#   clusters: (N,) array, should probably be integer type.  Cluster label for 
#       each point in pts
def dunn_index(pts, clusters):
    assert( len(pts) == len(clusters) )
    C = sorted( np.unique(clusters) )
    Z = []
    d_within = []
    for c in C:
        pts_c = pts[ clusters==c ]
        z = np.mean(pts_c, axis=0)
        dists_c = []
        for p in pts_c:
            dists_c.append( euclidean(p,z) )
        d_within.append( np.mean(dists_c) )
        Z.append(z)
    
    min_inter_d = np.inf
    for i,z1 in enumerate(Z):
        for z2 in Z[ i+1 : ]:
            d = euclidean(z1, z2)
            if d < min_inter_d:
                min_inter_d = d

    dunn_index = min_inter_d / np.max(d_within)
    return dunn_index

def get_dists(nearest_dict, data, N):
    dists = []
    for k,v in nearest_dict.items():
        for pt in v[:N]:
            dists.append( euclidean(data[k],pt) )
    return dists

def get_pairwise_dists(nearest_dict, data, N):
    dists = []
    for k,v in nearest_dict.items():
        pts = [data[k]] + v[:N]
        for i,pt1 in enumerate(pts):
            for pt2 in pts[i+1:]:
                dists.append( euclidean(pt1,pt2) )
    return dists

def make_nearest_dict(num_starts, n_range, data):
    if num_starts==0:
        return {}
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

def make_wave_data(num_waves, num_pts, noise_sigma, offset):
    x = np.linspace(0, 6*np.pi, num_pts)
    y = []
    for i in range(num_waves):
        y.append( np.sin(x) + i*offset )
    data = np.zeros((num_waves*num_pts,2))
    for i in range(num_waves):
        data[i*num_pts : (i+1)*num_pts, 0] = x
        data[i*num_pts : (i+1)*num_pts, 1] = y[i]
    noise = np.random.normal(0, noise_sigma, data.shape)
    data += noise
    return data


