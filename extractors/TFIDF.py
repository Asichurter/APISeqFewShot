import os
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

from utils.file import loadJson, dumpJson


##############################################
# 根据序列数据集(ngram,api)，先统计元素的样本内频率，
# 然后计算各个特征的TF-IDF值
##############################################
def getTFIDF(dataset_path,
                    dict_map_path,  # 将序列元素转化为一个实值的映射的路径，通常为wordMap.json
                    is_class_dir=True,
                    level='item',   # 在样本层次上上统计TFIDF还是类层次上
                    top_k=2000):    # 取tf-idf值最高的k个api

    value_map = loadJson(dict_map_path)
    value_min = min(value_map.values())
    value_max = max(value_map.values())
    value_size = value_max - value_min + 1

    frq_mat = []
    N = None

    for folder in tqdm(os.listdir(dataset_path)):

        if is_class_dir:
            items = os.listdir(dataset_path + folder + '/')
            assert N is None or N==len(items), "每个类内样本数量不一致！"
            N = len(items)

        else:
            items = [folder+'.json']

        for item in items:
            data = loadJson(dataset_path + folder + '/' + item)
            seqs = data['apis']

            #　映射token为实值
            seqs = [value_map[s] for s in seqs]

            hist, bins_sep = np.histogram(seqs,
                                        range=(value_min-0.5,value_max+0.5),   # 这样划分使得每一个int值都处于一个单独的bin中
                                        bins=value_size,
                                        normed=True)

            frq_mat.append(hist.tolist())

    frq_mat = np.array(frq_mat)

    # 如果要计算类级别的tfidf，则把类内样本的元素频率相加作为整个类的频率向量，
    # 然后在类的级别上计算tf和idf
    if level == 'class':
        frq_mat = frq_mat.reshape(-1,N,len(value_map))
        frq_mat = np.sum(frq_mat, axis=1)

    transformer = TfidfTransformer()
    transformer.fit(frq_mat)

    tf = np.mean(frq_mat, axis=0)

    tfidf = tf*transformer.idf_
    # 取tf-idf最大的k个api的下标
    top_k_idxes = tfidf.argsort()[::-1][:top_k]
    api_list = list(value_map.keys())

    top_k_apis = [api_list[i] for i in top_k_idxes]

    print("- Done -")
    return top_k_apis
    # return tfidf, transformer

if __name__ == '__main__':
    getTFIDF(dataset_path='/home/asichurter/datasets/JSONs/virushare-10-3gram/all/',
                                  dict_map_path='/home/asichurter/datasets/JSONs/virushare-10-3gram/data/wordMap.json',
                                  is_class_dir=True,
                                  level='item')
