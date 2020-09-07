import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from utils.color import getRandomColor
from utils.manager import PathManager
from utils.file import dumpJson, loadJson, dumpIterable
from scripts.embedding import aggregateApiSequences

k = 5
n = 5
qk = 15
N = 20
WORK_SPACE = ""

def apiCluster(dict_path, map_dump_path):
    api_mat = np.load(dict_path, allow_pickle=True)

    # pca = TSNE(n_components=2)
    # de_api_mat = pca.fit_transform(api_mat)
    # colors = getRandomColor(26, more=False)

    print("Clustering...")
    km = KMeans(n_clusters=26).fit(api_mat)
    km_wrapper = {i:int(c) for i,c in enumerate(km.labels_)}
    dumpJson(km_wrapper, map_dump_path)

    # plt.figure(figsize=(15,12))

    # for i,item in enumerate(de_api_mat):
    #     plt.scatter(item[0], item[1], color=colors[km.labels_[i]])

    # plt.scatter(de_api_mat[:,0], de_api_mat[:,1])
    # plt.show()


###############################################################
# 利用API聚类结果，API下标映射和转化后的字符串序列，将每个样本转化为A-Z的
# 字符序列，便于运行MSA。最终生成的是一个json文件，该文件中同类样本相邻
###############################################################
def convertApiCategory(clst_path, wrod_map_path, json_path, str_dump_path, max_len=300):
    word_map = loadJson(wrod_map_path)
    cluster_map = loadJson(clst_path)
    seqs = aggregateApiSequences(json_path, is_class_dir=True)

    str_mat = []
    for seq in seqs:
        seq = seq[:max_len]
        s = ""
        for idx in seq:
            api_idx = str(word_map[idx])
            s += chr(65+cluster_map[api_idx])

        str_mat.append(s)

    dumpIterable(str_mat, title="strings", path=str_dump_path)


def genFamilyProtoByMSA(str_path, proto_dump_path):



def runMSA()



if __name__ == '__main__':
    mng = PathManager("virushare-20-original")
    apiCluster(mng.WordEmbedMatrix(), mng.DataRoot()+"CategoryMapping.json")
    convertApiCategory(clst_path=mng.DataRoot()+"CategoryMapping.json",
                       wrod_map_path=mng.WordIndexMap(),
                       json_path=mng.Folder(),
                       str_dump_path=mng.DataRoot()+"CategorizedStringData.json")