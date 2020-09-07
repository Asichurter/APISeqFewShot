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
def aggregateApiSequences(path, is_class_dir=True):

    seqs = []

    for folder in tqdm(os.listdir(path)):
        folder_path = path + folder + '/'

        if is_class_dir:            # 如果是类文件夹，则整个路径下都是需要检索的JSON
            items = os.listdir(folder_path)
        else:                       # 如果是个体文件夹，路径下只有 文件夹名+.JSON 才是需要检索的
            items = [folder + '.json']

        for item in items:
            try:
                report = loadJson(folder_path + item)
                apis = report['apis']

                if len(apis) > 0:
                    seqs.append(apis)
            except Exception as e:
                print(folder, item, e)
                exit(-1)

    return seqs

##########################################################
# 根据收集的序列训练W2V模型，获得嵌入矩阵和词语转下标表，用于将API序列
# 下标化并嵌入为数值向量，两者存为文件形式。
##########################################################
def trainW2Vmodel(seqs, sg=0, size=64, min_count=1, cbow_mean=1,
                  save_matrix_path=None,
                  save_word2index_path=None,
                  padding=True):                # 是否在W2V转换矩阵中添加一个pad嵌入

    printBulletin('Traning Word2Vector...')
    model = Word2Vec(seqs, size=size, sg=sg, min_count=min_count, cbow_mean=cbow_mean)

    printBulletin('Saving...')
    matrix = model.wv.vectors
    word2index = {}

    if padding:
        pad_matrix = np.zeros((1, model.wv.vectors.shape[1]))
        matrix = np.concatenate((pad_matrix, matrix), axis=0)

        for i, w in enumerate(model.wv.index2word):
            word2index[w] = i + 1 if padding else i  # 由于idx=0要留给padding，因此所有的下标都加1
        word2index['<PAD>'] = 0

    if save_matrix_path:
        np.save(save_matrix_path, matrix)

    if save_word2index_path:
        dumpJson(word2index, save_word2index_path)

    if save_matrix_path is None and save_word2index_path is None:
        return matrix, word2index

    printBulletin('Done')

if __name__ == '__main__':
    manager = PathManager(dataset='virushare-20-original', d_type='all')

    # print(manager.FileData())

    seqs = aggregateApiSequences(manager.Folder())
    trainW2Vmodel(seqs,
                  save_matrix_path=manager.WordEmbedMatrix(),
                  save_word2index_path=manager.WordIndexMap(),
                  size=64)