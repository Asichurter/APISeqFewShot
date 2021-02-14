# -*- coding: utf-8
import sys
sys.path.append('../')

from preliminaries.dataset import makeDataFile
from utils.manager import PathManager
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('-l', '--maxSeqLen', type=int)

args = parser.parse_args()
dataset = args.dataset
max_seq_len = args.maxSeqLen


for d_type in ['train', 'validate', 'test']:
    manager = PathManager(dataset=dataset, d_type=d_type)

    makeDataFile(json_path=manager.Folder(),
                 w2idx_path=manager.WordIndexMap(),
                 seq_length_save_path=manager.FileSeqLen(),
                 data_save_path=manager.FileData(),
                 idx2cls_mapping_save_path=manager.FileIdx2Cls(),
                 num_per_class=20,
                 max_seq_len=max_seq_len,
                 verbose=False)
