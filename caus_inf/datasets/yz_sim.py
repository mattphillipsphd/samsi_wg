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


# 'batch' is a list of whatever comes out of the dataset
def seq_collate_fn(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    tts = [b[2] for b in batch]
    pids = [b[3] for b in batch]
    xs = torch.cat( [x.unsqueeze(1) for x in xs], axis=1 )
    ys = torch.cat( [y.unsqueeze(1) for y in ys], axis=1 )
    tts = torch.cat( [t.unsqueeze(1) for t in tts], axis=1 )
    pids = torch.LongTensor(pids)
    return (xs, ys, tts, pids)

def get_YZSim_loaders(cfg, modes=["train", "valid"]):
    ds_kwargs = OrderedDict()
    for k in []:
        ds_kwargs[k] = cfg[k]
    dl_kwargs = { "num_workers" : cfg["num_workers"],
            "batch_size" : cfg["batch_size"],
            "collate_fn" : seq_collate_fn }

    if cfg["cuda"] >= 0:
        dl_kwargs["pin_memory"] = True
    loaders = []

    if "train" in modes:
        train_dataset = YZSim(mode="train", **ds_kwargs)
        loaders.append( DataLoader(train_dataset, shuffle=True, **dl_kwargs) )
    if "valid" in modes:
        valid_dataset = YZSim(mode="valid", **ds_kwargs)
        loaders.append( DataLoader(valid_dataset, shuffle=False, **dl_kwargs) )
    if "test" in modes:
        test_dataset = YZSim(mode="test", **ds_kwargs)
        loaders.append( DataLoader(test_dataset, shuffle=False, **dl_kwargs) )
    return loaders

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
        Y2 = self._data_dict[pid]["Y"]
        treat_time = self._data_dict[pid]["treat"]
        Y = np.zeros( (len(Y2),) )
        Y[:treat_time] = Y2[:treat_time, 0]
        Y[treat_time:] = Y2[treat_time:, 1]
        tt = torch.zeros( len(X), 1 )
        tt[treat_time] = 1
        return torch.FloatTensor(X),torch.FloatTensor(Y),tt,pid

    def __len__(self):
        return len(self._pids)

    def get_input_size(self):
        return 100

    def get_pid(self, index):
        return self._pids[index]

    def _get_data_and_labels(self):
        if self._mode=="train":
            path = pj(self._data_supdir, "TP_train100.csv") # TODO
        elif self._mode=="valid" or self._mode=="test":
            path = pj(self._data_supdir, "TP_train100.csv") # TODO
#            path = pj(self._data_supdir, "TP_test.csv")
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
    x,y,tt,pid = dataset[4]
    print(x.shape)
    print(y.shape)
    print(tt.shape)
    print(pid)
    loader, = get_YZSim_loaders(cfg, modes="train")
    for data in loader:
        print("batch: ", len(data))
        for x in data:
            print("\t", x.shape)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=-1)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--cuda", type=int, default=0, help="-1 for CPU")
    cfg = vars( parser.parse_args() )
    main(cfg)

