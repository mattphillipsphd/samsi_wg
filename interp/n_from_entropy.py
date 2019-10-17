"""
Calculate the best N for UMAP using local entropy
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

def make_data(cfg):
    x = np.linspace(0, 6*np.pi, 200)
    y = []
    for i in range(4):
        y.append( np.sin(x) + i*0.25 )
    data = np.zeros((800,2))
    for i in range(4):
        data[i*200 : (i+1)*200, 0] = x
        data[i*200 : (i+1)*200, 1] = y[i]
    noise = np.random.normal(0, cfg["sigma"], data.shape)
    data += noise
    return data

def main(cfg):
    data = make_data(cfg)
#    plt.plot(data[:,0], data[:,1])
#    plt.show()
    for n in range(1,40):


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add argument("--num-waves", type=int, default=4)
    cfg = vars( parser.parse_args() )
    main(cfg)
