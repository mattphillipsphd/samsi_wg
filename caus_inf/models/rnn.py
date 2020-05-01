"""
RNN for clinical data
"""

import argparse
import numpy as np
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device=-1):
        super().__init__()
        self._device = device if device >= 0 else "cpu"
        self._hidden_size = hidden_size
        sk = np.sqrt(1/hidden_size)
        self._W_ih = torch.FloatTensor(input_size, hidden_size)\
                .uniform_(-sk,sk).to(device)
        self._W_hh = torch.FloatTensor(hidden_size, hidden_size)\
                .uniform_(-sk,sk).to(device)
        self._b_ih = torch.FloatTensor(hidden_size)\
                .uniform_(-sk,sk).to(device)
        self._b_hh = torch.FloatTensor(hidden_size)\
                .uniform_(-sk,sk).to(device)

    def forward(self, x):
        x = self._W_ih*x + self._b_ih + self._W_hh*h + self._b_hh
        return torch.tanh(x)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, batch_first=False,
            device=-1, do_manual=False):
        super().__init__()
        self._device = device if device >= 0 else "cpu"
        self._hidden_size = hidden_size

        self._rnn = MyRNN(input_size, hidden_size) if do_manual \
                else nn.GRUCell(input_size, hidden_size, bias=bias)
        self._linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden = torch.zeros(x.size(1), self._hidden_size, device=self._device)
        for x_step in x:
            hidden = self._rnn(x_step, hidden)
        output = hidden
        x = self._linear(output)
#        print(torch.min(x).item(), torch.max(x).item(), torch.mean(x).item(), torch.median(x).item())
        out = torch.sigmoid(x)
#        out = x
#        print(out.shape)
        return out

    def get_name(self):
        return "SimpleRNN"


def main(cfg):
    m_arglist = ["input_size", "hidden_size", "bias", "do_manual"]
    m_args = OrderedDict()
    for k in m_arglist:
        m_args[k] = cfg[k]
    model = SimpleRNN(**m_args)
    seq_len = cfg["test_seq_len"] if cfg["test_seq_len"]>0 \
            else cfg["max_length"]
    x = torch.FloatTensor(seq_len, cfg["batch_size"],
            cfg["input_size"]).uniform_(0,1)
    print("x: ", x.size())
    y = model(x)
    print("y: ", y.size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input-size", type=int, default=100)
    parser.add_argument("-h", "--hidden-size", type=int, default=256)
    parser.add_argument("--no-bias", dest="bias", action="store_false")
    parser.add_argument("--do-manual", action="store_true")
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument("-m", "--max-num-codes", type=int, default=40)
    parser.add_argument("-s", "--sequence-length", type=int, default=20,
            help="Measured in time increments")
    parser.add_argument("--test-seq-len", type=int, default=-1)
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    cfg = vars( parser.parse_args() )
    main(cfg)


