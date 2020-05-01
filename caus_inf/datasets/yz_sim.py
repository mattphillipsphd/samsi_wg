"""
Dataloader for Yongli Zhang's simulated data
"""


import abc
import argparse
import csv
import logging
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

class YZSim(Dataset):
    def __init__(self, data_supdir=pj(HOME, "Datasets/EHRs/CausInfSim"),
            mode="train", use_small=True):
        super().__init__()
        self._data_supdir = data_supdir
        self._data_dict = None
        self._labels = None
        self._mode = mode
        self._pids = None
        self._use_small = use_small

        self._get_data_and_labels()

    def __getitem__(self, index):
        pid = self._pids[index]
        X = self._data_dict[pid]["X"]
        Y = self._data_dict[pid]["Y"]
        tt = torch.zeros( len(X), 1 )
        tt[ self._data_dict[pid]["treat"] ] = 1
        return torch.FloatTensor(X),torch.FloatTensor(Y),tt

    def __len__(self):
        return len(self._pids)

    def get_pid(self, index):
        return self._pids[index]

    def _get_data_and_labels(self):
        if self._mode=="train":
            path = pj(self._data_supdir, "TP_train100.csv") # TODO
        elif self._mode=="test":
            path = pj(self._data_supdir, "TP_test.csv")
        else:
            raise NotImplementedError(self._mode)
        df = pd.read_csv(path)
        x_keys = [x for x in df.keys() if x.startswith("X")]
        y_keys = [y for y in df.keys() if y.startswith("Y")]
        
        self._data_dict = OrderedDict()
        self._pids = list( df["patient_id"].unique() )
        for pid in self._pids:
            df_p = df[ df["patient_id"] == pid ]
            self._data_dict[pid] = {}
            self._data_dict[pid]["X"] = np.array( df_p[x_keys] )
            self._data_dict[pid]["Y"] = np.array( df_p[y_keys] )
            self._data_dict[pid]["treat"] = list( df_p["treatment_time"] )[0]
            self._data_dict[pid]["ts"] = np.array( df_p["datetime"] )
    

def main(cfg):
    dataset = YZSim()
    x,y,tt = dataset[4]
    print(x.shape)
    print(y.shape)
    print(tt.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=-1)
    cfg = vars( parser.parse_args() )
    main(cfg)

