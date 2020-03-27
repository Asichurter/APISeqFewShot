import numpy as np
from functools import reduce

##########################################
# 将numpy矩阵内部的元素逐个相乘约减，类似于np.sum
##########################################
def matMulReduce(mat):
    mat = np.ravel(mat).tolist()
    multerm = reduce(lambda x,y: x*y, mat)
    return multerm


def batchDot(x, y,
             transpose=False):      # 是否需要对第二个元素进行转置运算
    checked_dim = 1 if not transpose else 2
    assert x.size(0)==y.size(0) and x.size(2)==y.size(checked_dim)

    if not transpose:
        y = y.transpose(1,2).contiguous()

    x_repeat = y.size(1)
    y_repeat = x.size(1)

    x = x.unsqueeze(2).repeat((1,1,x_repeat,1))
    y = y.unsqueeze(1).repeat((1,y_repeat,1,1))

    return (x*y).sum(dim=3)

