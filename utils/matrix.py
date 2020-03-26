import numpy as np
from functools import reduce

##########################################
# 将numpy矩阵内部的元素逐个相乘约减，类似于np.sum
##########################################
def matMulReduce(mat):
    mat = np.ravel(mat).tolist()
    multerm = reduce(lambda x,y: x*y, mat)
    return multerm