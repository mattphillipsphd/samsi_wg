"""
Trainer for training RNN
"""

import argparse
import logging
import numpy as np
import os
import shutil
import sys
from collections import OrderedDict

import torch
import torchvision as tv
import torch.nn.functional as F
from torch.distributions import Bernoulli, Uniform
from torch.utils.data import DataLoader, Dataset

from general.utils import make_or_get_session_dir
from pytorch.pyt_utils.utils import find_lr, get_summary_writer, \
        print_var_stats, save_model_pop_old

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.yz_sim import get_YZSim_loaders
from models.rnn import SimpleRNN

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_iter_ct = None

g_writer = None
def g_set_writer(writer):
    global g_writer
    g_writer = writer
def get_writer():
    return g_writer

g_writer_mode = None # E.g., train or test
def g_set_writer_mode(writer_mode):
    global g_writer_mode
    g_writer_mode = writer_mode


def get_model(input_size, cfg):
    print("Loading model with input size %d..." % input_size)
    model = SimpleRNN(input_size, cfg["rnn_hidden_size"], # TODO
            device=cfg["cuda"])
    cudev = "cpu" if cfg["cuda"] < 0 else cfg["cuda"]
    model = model.to(cudev)
    print("Done, loaded model %s" % model.get_name())
    return model

def train(model, loaders, cfg):
    cudev = "cpu" if cfg["cuda"] < 0 else cfg["cuda"]
    global g_iter_ct
    writer = get_writer()
    train_loader,valid_loader = loaders
    criterion = torch.nn.MSELoss(reduction="mean")
#    criterion = torch.nn.BCELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    g_iter_ct = 1
    for epoch in range( cfg["max_num_epochs"] ):
        g_set_writer_mode("train")
        for data in train_loader:
            x,y,tt,pid = data
            x = x.to(cudev)
            y = y.to(cudev)
            tt = tt.to(cudev)
            # x: (160, 32, 100)
            # y: (160, 32)
            # tt: (160, 32, 1)
#            print("x", x.shape)
#            print("y", y.shape)
#            print("tt", tt.shape)
#            raise
            xt = torch.cat([x,tt], dim=2)
            y = y[-1, :]

            yhat = model(xt).view(-1)
            
            optimizer.zero_grad()
            loss = criterion(yhat, y)
            writer.add_scalars("Loss", {g_writer_mode : loss.item()},g_iter_ct)
            print("Training loss: %f" % loss.item())
            g_iter_ct += 1

        with torch.no_grad():
            g_set_writer_mode("test")
            torch.cuda.empty_cache()

            test_loss = 0
            for ct,data in enumerate(valid_loader):
                x,y,tt,pid = data
                x = x.to(cudev)
                y = y.to(cudev)
                tt = tt.to(cudev)
                xt = torch.cat([x,tt], dim=2)
                y = y[-1, :]

                yhat = model(xt).view(-1)
                
                test_loss = criterion(yhat, y)
                g_iter_ct += 1
            writer.add_scalars("Loss", {g_writer_mode : test_loss}, g_iter_ct)
            print("Valid loss: %f" % test_loss.item())

def main(cfg):
    cfg["session_dir"] = make_or_get_session_dir(cfg["sessions_supdir"],
            cfg["model"], cfg["dataset"], cfg["resume_path"])
    writer = get_summary_writer(cfg["session_dir"])
    g_set_writer(writer)
    logging.basicConfig(filename=pj(cfg["session_dir"], "session.log"),
            level=logging.INFO, filemode="a")
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    s = "Session directory: %s" % cfg["session_dir"]
    logger.info(s)
    cfg["output_dir"] = pj(cfg["session_dir"], "eval")
    if not pe(cfg["output_dir"]):
        os.makedirs(cfg["output_dir"])
    loaders = get_YZSim_loaders(cfg)
    input_size = loaders[0].dataset.get_input_size()
    model = get_model(input_size+1, cfg) # TODO +1 for treatment var
    train(model, loaders, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input-file", type=str,
            default=pj(HOME, "Datasets/EHRs/CausInfSim/TP_train100.csv"))
    parser.add_argument("--sessions-supdir", type=str,
            default=pj(HOME, "Training/caus_inf/sessions"))
    parser.add_argument("--resume-path", type=str, default="",
            help="Path to model to resume training.  Should be in " \
                    "session_<n>/models")
    parser.add_argument("-m", "--model", default="simple_rnn",
            choices=["simple_rnn"])
    parser.add_argument("-d", "--dataset", default="YZSim",
            choices=["YZSim"])

    parser.add_argument("--train-valid-split", type=float, default=0.85)
    parser.add_argument("-s", "--sequence-length", type=int, default=20,
            help="Measured in time increments")
    parser.add_argument("-t", "--time-increment", type=int, default=30)
    parser.add_argument("-h", "--rnn-hidden-size", type=int, default=256)

    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float,
            default=0.001)

    parser.add_argument("-n", "--max-num-epochs", type=int, default=1000)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--cuda", type=int, default=0, help="-1 for CPU")
    cfg = vars( parser.parse_args() )
    main(cfg)

