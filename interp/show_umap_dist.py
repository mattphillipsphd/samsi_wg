"""
Show Dunn Index of UMAP clustering as function of n
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import umap
from scipy.spatial.distance import euclidean

sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
import wg_utils as U

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

def main(cfg):
    data = U.make_wave_data(cfg["num_waves"], cfg["num_points"], cfg["sigma"])
    data_name = "waves(4)"
    labels = []
    for c in range( cfg["num_waves"] ):
        labels += [c] * cfg["num_points"]

    ns = list(range(2, cfg["n_range"]))
    dis = []
    for n in ns:
        emb = umap.UMAP(n_neighbors=n, min_dist=cfg["min_dist"],
                metric=cfg["umap_metric"]).fit_transform(data)
        dis.append( U.dunn_index(emb, labels) )
    best_idx = np.argmax(dis)
    print("Best n: %d" % ns[best_idx])
    plt.plot(ns, dis)
    plt.xlabel("# nearest neighbors")
    plt.ylabel("Dunn Index")
    plt.title("Dunn Index on %s" % data_name)
    if cfg["show_plot"]:
        plt.show()
    if not pe(cfg["output_dir"]):
        os.makedirs( cfg["output_dir"] )
    plt.savefig( pj(cfg["output_dir"], "umap_dunn_index.png") )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-range", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--umap-metric", type=str, default="correlation")
    parser.add_argument("-o", "--output-dir", type=str,
            default=pj(HOME, "Repos/mattphillipsphd/samsi_wg/interp/output"))
    parser.add_argument("--show-plot", action="store_true")

    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--num-waves", type=int, default=4)
    
    cfg = vars( parser.parse_args() )
    main(cfg)

