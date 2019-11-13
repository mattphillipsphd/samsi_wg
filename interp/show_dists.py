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
from unidip import UniDip

sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
from interp.show_umap_dist import get_embeddings
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
    data,labels = U.make_wave_data(cfg["num_waves"], cfg["num_points"],
            cfg["sigma"], cfg["offset"], rand_seed=cfg["seed"])
    nearest_dict = U.make_nearest_dict(cfg["num_starts"], cfg["n_range"], data)
    prop_dict = OrderedDict()
    ns = range(2, cfg["n_range"]+1)
    prop_vals = []
    for n in ns:
        if cfg["property"] == "pwise-dist":
            dists = U.get_pairwise_dists(nearest_dict, data, n)
        elif cfg["property"] == "dist0_ff":
            dists = U.get_dists(nearest_dict, data, n)
            prop_vals.append( np.var(dists) / np.mean(dists) )
#            bins = np.linspace(0, np.quantile(dists, 0.99), cfg["num_bins"])
#            h,_ = np.histogram(dists, bins=bins)
#            h = h/np.sum(h)
#            intervals = UniDip(h).run()
#            prop_vals.append( len(intervals) )
        else:
            raise NotImplementedError( cfg["property"] )
        prop_dict[n] = dists        

    _,dunn_idxs = get_embeddings(data, labels, ns, cfg)

    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("# nearest neighbors")
    ax1.set_ylabel("Dunn Index", color=color)
    ax1.plot(ns, dunn_idxs, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel(cfg["property"], color=color)
    ax2.plot(ns, prop_vals, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Dunn Index and property %s" % cfg["property"])
    if not pe(cfg["output_dir"]):
        os.makedirs( cfg["output_dir"] )
    plt.savefig( pj(cfg["output_dir"], "umap_di_and_props.png") )
    if cfg["show_plot"]:
        plt.show()
    plt.close()
    
    plot_dists(prop_dict, cfg["num_bins"])
    plt.savefig( pj(cfg["output_dir"], "umap_%s.png" % cfg["property"]) )
    if cfg["show_plot"]:
        plt.suptitle("Distribution '%s' as function of n" % cfg["property"])
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--num-waves", type=int, default=4)
    parser.add_argument("--num-starts", type=int, default=10)
    parser.add_argument("--offset", type=float, default=0.5)
    parser.add_argument("--n-range", type=int, default=20)
    parser.add_argument("--num-bins", type=int, default=50)
    
    parser.add_argument("-o", "--output-dir", type=str,
            default=pj(HOME, "Repos/mattphillipsphd/samsi_wg/interp/output"))
    parser.add_argument("--show-plot", action="store_true")


    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="euclidean")
    parser.add_argument("--umap-init", type=str, default="spectral")

    parser.add_argument("--property", default="pwise-dist",
            choices=["pwise-dist", "dist0_ff"])

    cfg = vars( parser.parse_args() )
    main(cfg)

