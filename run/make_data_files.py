# -*- coding: utf-8

from preliminaries.dataset import makeDataFile
from utils.manager import PathManager

dataset_name = '...'
item_per_family = 20
max_seq_len = 300

for d_type in ['train', 'validate', 'test']:
    manager = PathManager(dataset=dataset_name, d_type=d_type)

    makeDataFile(json_path=manager.Folder(),
                 w2idx_path=manager.WordIndexMap(),
                 seq_length_save_path=manager.FileSeqLen(),
                 data_save_path=manager.FileData(),
                 idx2cls_mapping_save_path=manager.FileIdx2Cls(),
                 num_per_class=item_per_family,
                 max_seq_len=max_seq_len)

