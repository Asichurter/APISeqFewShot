import numpy as np
import torch as t
import torch.nn.functional as F

from scripts.dataset import makeDataFile
from utils.manager import PathManager
from scripts.reshaping import makeMatrixData
from scripts.preprocessing import apiStat
from models.ProtoNet import IncepProtoNet
from utils.matrix import batchDot

# 制作基于下标的数据集
################################################################
# for d_type in ['train', 'validate', 'test']:
#     manager = PathManager(dataset='virushare_20', d_type=d_type)
#
#     makeDataFile(json_path=manager.Folder(),
#                  w2idx_path=manager.WordIndexMap(),
#                  seq_length_save_path=manager.FileSeqLen(),
#                  data_save_path=manager.FileData(),
#                  num_per_class=20,
#                  max_seq_len=4000)
################################################################


# 统计序列长度分布
################################################################
# apiStat('D:/peimages/PEs/virushare_20/jsons/',
#         ratio_stairs=[500, 1000, 2000, 4000, 5000, 10000, 20000, 50000],
#         class_dir=False)
################################################################


# 转化数据集
################################################################
# manager = PathManager(dataset='virushare_20', d_type='train')
# matrix = np.load(manager.WordEmbedMatrix(), allow_pickle=True)
# mapping = get1DRepreByRuduc(matrix)
# seq = t.load(manager.FileData())
# seq = reshapeSeqToMatrixSeq(seq, mapping, flip=True)
# t.save(t.Tensor(seq), manager.FileData())
# makeMatrixData(dataset='virushare_20', shape=(40, 10, 10))

# 翻转序列
################################################################
# a[:,1::2] = np.apply_along_axis(lambda x: np.flipud(x), 2, a[:,1::2])
################################################################


# 测试Inception模型
################################################################
# m = IncepProtoNet(channels=[1,32,1],
#                   depth=3)
# s = t.randn((5, 5, 100, 8, 5))
# q = t.randn((75, 100, 8, 5))
# out = m(s, q)
################################################################

# 测试自注意力
################################################################
x = t.Tensor([
    [
        [1,2,3],
        [3,4,5]
    ],
    [
        [5, 6, 7],
        [7, 8, 9]
    ]
])

Q = t.nn.Linear(3,3, bias=False)
K = t.nn.Linear(3,3, bias=False)
V = t.nn.Linear(3,3, bias=False)

t.nn.init.constant_(Q.weight, 1)
t.nn.init.constant_(K.weight, 1)
t.nn.init.constant_(V.weight, 1)

k = K(x)
q = Q(x)
v = V(x)

w = batchDot(q, k, transpose=True)
p = t.softmax(w, dim=2)
z = batchDot(p, v, transpose=False)
################################################################
