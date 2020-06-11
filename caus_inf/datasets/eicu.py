"""
Dataloader for eICU dataset.  This assumes we're using eicu data as
preprocessed according to Manduchi et al. 2019, i.e.

https://github.com/mattphillipsphd/dpsom.git
"""


import abc
import argparse
import csv
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import platform
import sys
import torch
import torchvision as tv

from collections import OrderedDict
from PIL import Image
from skimage.transform import resize

from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ci_utils as CU

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

class EicuDS(Dataset):
    def __init__(self, data_supdir=pj(HOME, "Datasets/eicu-2.0")):
        super().__init__()
        self._batch_to_lst = None
        self._data_supdir = data_supdir
        self._pid_to_batch = None
        self._pids = None
        self._var_keys = None

        self._load_pids_and_bdicts()
        self._make_var_keys()

    def __getitem__(self, index):
        """
        Return a dict with patient id, variables, labels, timestamps
        """
        d = {}
        pid = self._pids[index]
        d["pid"] = pid
        d["variables"] = self._get_variables(pid)
        labels,ts = self._get_labels_and_ts(pid)
        d["labels"] = labels
        d["ts"] = ts
        return d

    def __len__(self):
        return len( self._pids )

    def get_pid(self, index):
        return self._pids[index]

    def get_var_index(self, var_key):
        return self._var_keys.index(var_key)

    def _get_labels_and_ts(self, pid):
        bnum = self._pid_to_batch[pid]
        batch_path = pj(self._data_supdir, "labels", "batch_%d.h5" % bnum)
        batch = pd.read_hdf(batch_path)
        pt_data = batch[ batch["patientunitstayid"]==pid ]
        labels = torch.FloatTensor( np.array( pt_data[ CU.g_pt_endpoints ] ) )
        ts = torch.FloatTensor( np.array( pt_data["ts"] ) )
        return labels,ts


    def _get_variables(self, pid):
        bnum = self._pid_to_batch[pid]
        batch_path = pj(self._data_supdir, "time_grid", "batch_%d.h5" % bnum)
        batch = pd.read_hdf(batch_path)
        variables = batch[ batch["patientunitstayid"]==pid ]
        variables = torch.FloatTensor( np.array( variables[self._var_keys] ) )
        return variables


    def _load_pids_and_bdicts(self):
        self._pids = []
        with open( pj(self._data_supdir, "included_pid_stays.txt") ) as fp:
            for line in fp:
                self._pids.append( int( line.strip() ) )

        bdicts = np.load( pj(self._data_supdir, "patient_batches.pickle") )
        self._batch_to_lst = bdicts["batch_to_lst"]
        self._pid_to_batch = bdicts["pid_to_batch"]

    def _make_var_keys(self):
        def _read_list(path):
            v_list = []
            with open(path) as fp:
                for line in fp:
                    v_list.append( line.strip() )
            return v_list

        self._var_keys = []
        for vcat in ["vitalPeriodic", "vitalAperiodic", "lab"]:
            inc_path = pj(self._data_supdir, CU.g_includes[vcat])
            v_list = _read_list(inc_path)
            prefix = CU.g_prefixes[vcat]
            self._var_keys += [prefix + "_" + v for v in v_list]

        self._var_keys.remove("lab_-bands") # Necessary apparently
    

def main(cfg):
    output_dir = os.path.abspath( cfg["output_dir"] )
    if not pe(output_dir):
        os.makedirs(output_dir)
    ds = EicuDS()
    print("Number of patients: %d" % len(ds))
    index = cfg["patient_index"]
    d = ds[index]

    t = d["ts"]
    y = d["variables"][ :, ds.get_var_index("vs_heartrate") ]

    pid = ds.get_pid(index)
    plt.plot(t,y)
    plt.xlabel("Time (h)")
    plt.ylabel("vs_heartrate")
    plt.title("Patient %d" % pid)
    plt.savefig( pj(output_dir, "%d_hr" % pid) )
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str,
            default=pj(HOME, "Training/caus_inf/test_out/EicuDS"))
    parser.add_argument("-i", "--patient-index", type=int, default=0)
    cfg = vars( parser.parse_args() )
    main(cfg)

