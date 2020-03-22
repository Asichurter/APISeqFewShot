
import os
import numpy as np
from tqdm import tqdm

from utils.file import loadJson, dumpJson
from utils.display import printBulletin
from utils.manager import PathManager

from gensim.models.word2vec import Word2Vec



############################################
# 本函数用于从已经处理好的json文件中收集所有样本的api
# 序列用于无监督训练嵌入。返回的是序列的列表。
############################################
def aggregateApiSequences(path):
    seqs = []

    for folder in tqdm(os.listdir(path)):
        folder_path = path + folder + '/'
        for item in os.listdir(folder_path):
            report = loadJson(folder_path + item)
            apis = report['apis']
            seqs.append(apis)

    return seqs

##########################################################
# 根据收集的序列训练W2V模型，获得嵌入矩阵和词语转下标表，用于将API序列
# 下标化并嵌入为数值向量，两者存为文件形式。
##########################################################
def trainW2Vmodel(seqs, sg=0, size=64, min_count=1, cbow_mean=1,
                  save_matrix_path=None, save_word2index_path=None):

    printBulletin('Traning Word2Vector...')
    model = Word2Vec(seqs, size=size, sg=sg, min_count=min_count, cbow_mean=cbow_mean)

    printBulletin('Saving...')
    if save_matrix_path:
        pad_matrix = np.zeros((1, model.wv.vectors.shape[1]))
        matrix = np.concatenate((pad_matrix, model.wv.vectors), axis=0)
        np.save(save_matrix_path, matrix)

    if save_word2index_path:
        word2index = {}
        for i, w in enumerate(model.wv.index2word):
            word2index[w] = i+1         # 由于idx=0要留给padding，因此所有的下标都加1
        dumpJson(word2index, save_word2index_path)

    printBulletin('Done')


if __name__ == '__main__':
    manager = PathManager(dataset='virushare_20')

    # print(manager.FileData())

    seqs = aggregateApiSequences(manager.FolderPath)
    trainW2Vmodel(seqs,
                  save_matrix_path='D:/peimages/JSONs/virushare_20/data/matrix.npy',
                  save_word2index_path='D:/peimages/JSONs/virushare_20/data/wordMap.json')