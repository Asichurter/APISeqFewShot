import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from utils.color import getRandomColor

k = 5
n = 5
qk = 15
N = 20
WORK_SPACE = ""

def apiCluster(dict_path):
    api_mat = np.load(dict_path, allow_pickle=True)

    pca = TSNE(n_components=2)#PCA(n_components=2)
    de_api_mat = pca.fit_transform(api_mat)
    colors = getRandomColor(26, more=False)

    km = KMeans(n_clusters=26).fit(api_mat)
    plt.figure(figsize=(18,15))

    for i,item in enumerate(api_mat):
        plt.scatter(item[0], item[1], color=colors[km.labels_[i]])

    # plt.scatter(de_api_mat[:,0], de_api_mat[:,1])
    plt.show()

if __name__ == '__main__':
    apiCluster("D:/datasets/virushare-20-3gram-tfidf/data/matrix.npy")