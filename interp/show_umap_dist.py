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
from collections import OrderedDict
from scipy.spatial.distance import euclidean

sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
import wg_utils as U

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def plot_embeddings(embeddings, labels, cfg):
    colors = ["r", "b", "g", "m", "c", "k", "y"]
    c_list = np.unique(labels)
    N = len(embeddings)
    figsize = (10,9)
    num_rows = int( np.floor(np.sqrt(N)) )
    num_cols = int( np.ceil(N/num_rows) )
    fig,axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    ct = 0
    for k,v in embeddings.items():
        i = ct // num_cols
        j = ct % num_cols
        ax = axs[i][j]
        for c,c_i in enumerate(c_list):
            ax.scatter(v[labels==c, 0], v[labels==c, 1], color=colors[c_i], s=4)
        ax.set_title("n = %d" % k)
        ct += 1

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    if not pe(cfg["output_dir"]):
        os.makedirs( cfg["output_dir"] )
    plt.savefig( pj(cfg["output_dir"], "umap_embeddings.png") )
    if cfg["show_plot"]:
        plt.show()
    plt.close()

def get_embeddings(data, labels, ns, cfg):
    dunn_idxs = []
    embeddings = OrderedDict()
    for n in ns:
        emb = umap.UMAP(n_neighbors=n, min_dist=cfg["min_dist"],
                metric=cfg["umap_metric"],
                init=cfg["umap_init"]).fit_transform(data)
        embeddings[n] = emb
        dunn_idxs.append( U.dunn_index(emb, labels) )
    return embeddings,dunn_idxs


def main(cfg):
    data,labels = U.make_wave_data(cfg["num_waves"], cfg["num_points"],
            cfg["sigma"], cfg["offset"], rand_seed=cfg["seed"])
    data_name = "waves(4)"

    ns = list(range(2, cfg["n_range"]+1))
    embeddings,dunn_idxs = get_embeddings(data, labels, ns, cfg)
    plot_embeddings(embeddings, labels, cfg)
    best_idx = np.argmax(dunn_idxs)
    print("Best n: %d" % ns[best_idx])
    plt.plot(ns, dunn_idxs)
    plt.xlabel("# nearest neighbors")
    plt.ylabel("Dunn Index")
    plt.title("Dunn Index on %s" % data_name)
    if not pe(cfg["output_dir"]):
        os.makedirs( cfg["output_dir"] )
    plt.savefig( pj(cfg["output_dir"], "umap_dunn_index.png") )
    if cfg["show_plot"]:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--n-range", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="euclidean")
    parser.add_argument("--umap-init", type=str, default="spectral")
    parser.add_argument("-o", "--output-dir", type=str,
            default=pj(HOME, "Repos/mattphillipsphd/samsi_wg/interp/output"))
    parser.add_argument("--show-plot", action="store_true")

    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--num-waves", type=int, default=4)
    parser.add_argument("--offset", type=float, default=0.5)

    cfg = vars( parser.parse_args() )
    main(cfg)

