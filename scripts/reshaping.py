import numpy as np
import torch as t

from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.manager import PathManager
from utils.matrix import matMulReduce

def get1DRepreByRuduc(matrix):
    reduce = PCA(n_components=1)
    matrix = reduce.fit_transform(matrix).squeeze()

    # 将<PAD>一直置为0
    # TODO: <PAD>的值？
    matrix[0] = 0

    # 其他值线性投影至1-255内，int型
    max_ = max(matrix)
    min_ = min(matrix)
    for i,val in enumerate(matrix):
        if i!=0:
            matrix[i] = int(1 + (val - min_)/(max_ - min_)*254)

    return matrix


def reshapeSeqToMatrixSeq(seq, mapping, shape=(100, 8, 5), flip=True):
    if isinstance(seq, t.Tensor):
        seq = seq.numpy()
    seq = seq.astype(np.int)
    seq = np.array([[mapping[i] for i in s] for s in seq])

    assert matMulReduce(shape)==seq.shape[1], \
        '指定的shape % s与seq维度 %s 不一致！'%(str(shape), str(seq.shape))

    seq = seq.reshape((seq.shape[0], *shape))

    # 将每个数据2D矩阵的奇数行的前后顺序改变
    # 1,2,3,         1,2,3,
    # 4,5,6,    =>   6,5,4,
    # 7,8,9          7,8,9
    if flip:
        seq[:,:,1::2,:] = np.flip(seq[:,:,1::2,:], axis=3)

    return seq


def makeMatrixData(dataset):
    for d_type in tqdm(['train', 'validate', 'test']):
        manager = PathManager(dataset=dataset, d_type=d_type)
        matrix = np.load(manager.WordEmbedMatrix(), allow_pickle=True)
        mapping = get1DRepreByRuduc(matrix)
        seq = t.load(manager.FileData())
        seq = reshapeSeqToMatrixSeq(seq, mapping, flip=True)
        t.save(t.Tensor(seq), manager.FileData())