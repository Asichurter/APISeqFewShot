import os
import shutil
import random
import torch as t
import pandas as pd
from time import time
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from utils.file import loadJson, dumpJson
from utils.display import printState
from utils.manager import PathManager

magic = 7365055

##########################################################
# 本文件是为了将数据集中的类进行随机抽样移动，可以用于数据集分割
##########################################################
def splitDatas(src, dest, ratio, mode='x', is_dir=False):
    '''
    将生成的样本按比例随机抽样分割，并且移动到指定文件夹下，用于训练集和验证集的制作
    src:源文件夹
    dest:目标文件夹
    ratio:分割比例或者最大数量
    '''
    assert mode in ['c', 'x'], '选择的模式错误，只能复制c或者剪切x'

    All = os.listdir(src)
    size = int(len(All) * ratio) if ratio < 1 else ratio

    assert len(All) >= size, '分割时，总数量没有要求的数量大！'

    random.seed(time() % magic)
    samples_names = random.sample(All, size)
    num = 0
    for item in tqdm(All):
        if item in samples_names:
            num += 1
            path = src + item
            if mode == 'x':
                shutil.move(path, dest)
            else:
                if is_dir:
                    shutil.copytree(src=path, dst=dest+item)
                else:
                    shutil.copy(src=path, dst=dest)


##########################################################
# 本函数主要用于数据集的文件生成。
# 用于根据已经按类分好的JSON形式数据集，根据已经生成的嵌入矩阵和
# 词语转下标表来将数据集整合，token替换为对应的嵌入向量，同时pad，最后
# 将序列长度文件和数据文件进行存储的总调用函数。运行时要检查每个类的样本
# 数，也会按照最大序列长度进行截断。
##########################################################
def makeDataFile(json_path,
                 w2idx_path,
                 matrix_path,
                 seq_length_save_path,
                 data_save_path,
                 num_per_class,
                 max_seq_len=600):

    data_list = []

    printState('Loading config data...')
    word2index = loadJson(w2idx_path)
    embed_matrix = np.load(matrix_path, allow_pickle=True)

    printState('Read main data...')
    for cls_idx, cls_dir in tqdm(enumerate(os.listdir(json_path))):
        class_path = json_path + cls_dir + '/'

        assert num_per_class == len(os.listdir(class_path)), \
            '数据集中类%s的样本数量%d与期望的样本数量不一致！'%\
            (cls_dir, len(os.listdir(class_path)), num_per_class)

        for item in os.listdir(class_path):
            report = loadJson(class_path + item)
            apis = report['apis']
            data_list.append(apis)          # 添加API序列

        # label_list += [cls_idx] * num_per_class     # 添加一个类的样本标签

    printState('Converting...')
    data_list = convertApiSeq2DataSeq(data_list,
                                      word2index,
                                      embed_matrix,
                                      max_seq_len)      # 转化为嵌入后的数值序列列表

    seq_length_list = {i:len(seq) for i,seq in enumerate(data_list)}   # 数据的序列长度

    data_list = pad_sequence(data_list, batch_first=True, padding_value=0)  # 数据填充0组建batch

    printState('Dumping...')
    dumpJson(seq_length_list, seq_length_save_path)     # 存储序列长度到JSON文件
    t.save(data_list, data_save_path)                   # 存储填充后的数据文件

    printState('Done')

##########################################################
# 本函数是makeDataFile函数的调用函数，主要用于截断序列，同时根据下标
# 表和嵌入矩阵将token替换为嵌入向量。
##########################################################
def convertApiSeq2DataSeq(api_seqs, word2idx, matrix, max_size):
    data_seq = []

    for seq in api_seqs:
        appended_seq = []

        if len(seq) > max_size:         # 根据规定的最大序列长度截断序列
            seq = seq[:max_size]

        for i,api in enumerate(seq):
            appended_seq.append(matrix[word2idx[api]].tolist())     # 先将API token转化为下标，再从矩阵中拿出对应位置的嵌入向量
        data_seq.append(t.Tensor(appended_seq))

    return data_seq


if __name__ == '__main__':
    manager = PathManager(dataset='virushare_20', d_type='test')
    makeDataFile(json_path= manager.Folder(),
                 w2idx_path=manager.WordIndexMap(),
                 matrix_path=manager.WordEmbedMatrix(),
                 seq_length_save_path=manager.FileSeqLen(),
                 data_save_path=manager.FileData(),
                 num_per_class=20,
                 max_seq_len=600)
    # splitDatas(src='D:/peimages/JSONs/virushare_20/train/',
    #            dest='D:/peimages/JSONs/virushare_20/test/',
    #            ratio=15,
    #            mode='x',
    #            is_dir=True)

    # a = [t.Tensor([[1, 2], [3, 4], [5, 6]]), t.Tensor([[11, 12], [13, 14]]), t.Tensor([[21, 22]])]
    # data_write_csv('data.csv', a)