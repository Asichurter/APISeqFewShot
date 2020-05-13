from time import time
import random as rd

magic = 7355608
magic_list_size = 100000

def magicSeed():
    return time()%magic

def magicList():
    return [i for i in range(magic)]

def randomList(num, min_=0, max_=magic, seed=None, allow_duplicate=True):
    assert max_-min_ > num or allow_duplicate, '不允许重复时，范围必须大于采样数！'

    seed = magicSeed() if seed is None else seed

    rd.seed(seed)
    if allow_duplicate:
        return [rd.randint(min_, max_) for i in range(num)]
    else:
        rd_set = set()
        while len(rd_set) < num:
            rd_set.add(rd.randint(min_, max_))
        return list(rd_set)
