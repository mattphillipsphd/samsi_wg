"""
Produce the joing probability of diagnoses and treatment
"""

import argparse
import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
import time

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def get_pt_ids(data_dir):
    d = pj(os.path.dirname(data_dir), "static_pts")
    pt_ids = [os.path.splitext(f)[0] for f in os.listdir(d)]
    return pt_ids

def main(cfg):
    data_dir = os.path.abspath( cfg["data_dir"] )
    if cfg["max_N"] < 0:
        treat_df = pd.read_csv( pj(data_dir, "treatment.csv") )
        diag_df = pd.read_csv( pj(data_dir, "diagnosis.csv") )
    else:
        treat_df = pd.read_csv( pj(data_dir, "treatment.csv"),
                nrows=cfg["max_N"] )
        diag_df = pd.read_csv( pj(data_dir, "diagnosis.csv"),
                nrows=cfg["max_N"] )

    diag_df = diag_df.sort_values(by="patientunitstayid")
    treat_df = treat_df.sort_values(by="patientunitstayid")

    pt_ids = get_pt_ids(data_dir)
    diag_df = diag_df[ diag_df.patientunitstayid.isin(pt_ids) ]
    treat_df = treat_df[ treat_df.patientunitstayid.isin(pt_ids) ]

    diags = diag_df.diagnosisstring
    treats = treat_df.treatmentstring
    diag_toks = [x.split("|") for x in diags]
    treat_toks = [x.split("|") for x in treats]
    
    d1 = [x[0] for x in diag_toks]
    t1 = [x[0] for x in treat_toks]

    unique_d1s = sorted( np.unique(d1) )
    unique_t1s = sorted( np.unique(t1) )
    print("Diagnoses")
    print(unique_d1s)
    print("Treatments")
    print(unique_t1s)

    d1_d = { diag : i for i,diag in enumerate(unique_d1s) }
    t1_d = { treat : i for i,treat in enumerate(unique_t1s) }

    diag_idxs = np.array( [d1_d[diag] for diag in d1] )
    treat_idxs = np.array( [t1_d[treat] for treat in t1] )

    d_pids = diag_df.patientunitstayid
    t_pids = treat_df.patientunitstayid
    
    M = len(unique_d1s)
    N = len(unique_t1s)
    joint_counts,total_counts = np.zeros((N,M)), np.zeros((N,M))

    for d_pid in d_pids.unique():
        dp_idxs = np.array(d_pids==d_pid)
        tp_idxs = np.array(t_pids==d_pid)

        diags_p = diag_idxs[dp_idxs]
        treats_p = treat_idxs[tp_idxs]
        #print(len(diags_p), len(treats_p))

        for j in diags_p:
            for i in range(N):
                total_counts[i][j] += 1
                if i in treats_p:
                    joint_counts[i][j] += 1

    joint_prob = joint_counts/total_counts
    print(joint_prob)
    ax = plt.imshow(joint_prob, cmap="gray", interpolation="none", vmin=0.0,
            vmax=1.0)
    plt.gcf().colorbar(ax)
    plt.savefig( pj(data_dir, "treat_diag_joint_prob1.png") )
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str,
            default=pj(HOME, "Datasets/EHRs/PhysioNet/eICU/csv"))
    parser.add_argument("-N", "--max-N", type=int, default=1000,
            help="Set to -1 for all rows")
    cfg = vars( parser.parse_args() )
    main(cfg)

