import os
import argparse
import numpy as np
import random
import sys
import time
# from tqdm import tqdm
# import ipdb
# import pickle

# import torch
# import torch.nn as nn

from Configs.builder import get_configs
#<SETTINGS>
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)
#<INITIAL_ARGUEMENTS>
parser = argparse.ArgumentParser(description="Uneven Event Modeling")
    #DATASET
parser.add_argument(
    '-d', '--dataset_name', default='act', type=str, metavar='DATASET', help='dataset name',choices=['tvr', 'act']
)
    #GPU
parser.add_argument(
    '--gpu', default = '1', type = str, help = 'specify gpu device'
)
    #EVALUATION
parser.add_argument('--eval', action='store_true')
    #LOAD CHECKPOINT
parser.add_argument('--resume', default='', type=str)
args = parser.parse_args()



def main():
    DATASET_NAME = "TVR" #Can use parse
    cfg = get_configs(DATASET_NAME)
    print(cfg)

if __name__ == "__main__":
    main()