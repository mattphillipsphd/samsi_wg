"""
Calculate the best N for UMAP using local entropy
"""

import argparse
import numpy as np
import os

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

def main(cfg):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cfg = vars( parser.parse_args() )
    main(cfg)
