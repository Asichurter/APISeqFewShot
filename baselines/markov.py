import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt

from preliminaries.embedding import aggregateApiSequences
from utils.file import loadJson, dumpIterable, dumpJson
from utils.manager import PathManager
from baselines.alignment import apiCluster
from utils.timer import StepTimer
from utils.magic import sample, magicSeed, nRandom
from utils.stat import calBeliefeInterval

k = 10
n = 10
qk = 5
N = 20


def findOptK(dict_path, k_range=(2,50)):
    mat = np.load(dict_path)
    wss = []

    for k_ in tqdm(range(k_range[0],k_range[1]+1)):
        kmeans = KMeans(n_clusters=k_).fit(mat)
        wss.append(kmeans.inertia_)

    kaxis = np.arange(k_range[0],k_range[1]+1)

    plt.plot(kaxis, wss)
    plt.show()

#####################################################
# 将原json文件中的序列根据聚类结果替换为类簇序列，同时使用一个
# 最大长度截断，并保存为npy文件
#####################################################
def makeClusteredData(json_path, cluster_path, word_map_path, dump_path, max_len=1000):
    word_map = loadJson(word_map_path)
    cluster_map = loadJson(cluster_path)
    seqs = aggregateApiSequences(json_path, is_class_dir=True)

    mat = []
    for seq in seqs:
        seq = seq[:max_len]
        s = []
        for idx in seq:
            s.append(cluster_map[str(word_map[idx])])
        while len(s) < max_len:
            s.append(-1)

        mat.append(s)

    np.save(dump_path, np.array(mat))


#####################################################
# 给定一个类别中的所有序列，生成该类别的转换矩阵
#####################################################
def makeTranMatrix(seqs, n_cluster, maxlen=1000):
    matrix = np.zeros((n_cluster,n_cluster))

    for seq in seqs:
        for i in range(0,len(seq)-1):
            if seq[i+1] == -1:
                break
            x = seq[i]
            y = seq[i+1]
            matrix[x][y] += 1

    row_sum = matrix.sum(0)
    mask = row_sum==0
    np.putmask(row_sum,mask,1)      # 将行和为0的位置置为1，防止除0错误
    normalized_matrix = (matrix.T / row_sum).T

    return normalized_matrix

##################################################
# 根据生成的转换矩阵组，根据最大转换值将序列转换为组内类别的
# 序列
##################################################
def traverse(seq, matrices, maxlen=1000):
    res_seq = []

    for i in range(0, maxlen-1):
        if seq[i] == -1:
            res_seq.append(-1)
            continue

        prob = matrices[:,seq[i],seq[i+1]]
        res_seq.append(np.argmax(prob))             # 将所有类中该i->i+1状态转移最大的类下标加入

    return res_seq


#############################################
# 根据输入序列，在多个类的转换矩阵中进行累计加分，
# 返回该序列在所有类的类转换矩阵中的总得分
#############################################
def scoreSeqOnTranMats(seq, tran_mats):
    score = np.zeros((len(tran_mats)))
    for i in range(len(seq)-1):
        score += tran_mats[:,seq[i],seq[i+1]]
    return score

def scoreMarkovEpisode(clustered_data_path, epoch=300, n_cluster=10, maxlen=1000, verbose=True):
    acc_hist = []
    matrix = np.load(clustered_data_path)
    class_pool = list(range(len(matrix) // N))
    item_pool = set(range(N))
    tm = StepTimer(epoch)

    tm.begin()
    for i in range(epoch):
        if verbose:
            print("Epoch", i)
        supports = []
        queries = []

        task_seed = magicSeed()
        sampling_seed = magicSeed()
        class_seeds = nRandom(n, sampling_seed)

        label_space = sample(class_pool, n, task_seed)

        for cls,cseed in zip(label_space, class_seeds):
            support_idxes = sample(item_pool, k, cseed, return_set=True)
            query_idxes = sample(item_pool.difference(support_idxes), qk, cseed, return_set=True)

            support_idxes = np.array(list(support_idxes)) + N*cls
            query_idxes = np.array(list(query_idxes)) + N*cls

            supports += [matrix[i] for i in support_idxes]
            queries += [matrix[i] for i in query_idxes]

        supports = np.array(supports).reshape(n,k,-1)
        queries = np.array(queries).reshape(n,qk,-1)

        # 利用原数据的类簇转换序列，生成每个类的类簇转换矩阵
        cluster_tran_mats = []
        for cls in range(n):
            cluster_tran_mats.append(makeTranMatrix(supports[cls], n_cluster=n_cluster,maxlen=maxlen))
        cluster_tran_mats = np.stack(cluster_tran_mats, axis=0)

        # 利用每个类的类簇转换矩阵，根据其中状态转移的最大值，转换为类别之间的转换序列
        class_tran_seqs = []
        for cls in range(n):
            for support in supports[cls]:
                class_tran_seqs.append(traverse(support, cluster_tran_mats, maxlen))
        class_tran_seqs = np.stack(class_tran_seqs, axis=0).reshape(n,k,-1)

        # 根据类别之间的转换序列，生成每个类的类别转换矩阵
        class_tran_mats = []
        for cls in range(n):
            # 由于是类别之间的转换序列，因此类簇数量等于类别数量
            class_tran_mats.append(makeTranMatrix(class_tran_seqs[cls], n_cluster=n, maxlen=maxlen))
        class_tran_mats = np.stack(class_tran_mats, axis=0).reshape(n,n,n)

        query_class_tran_seqs = []
        for cls in range(n):
            for query in queries[cls]:
                query_class_tran_seqs.append(traverse(query, cluster_tran_mats, maxlen))

        acc_count = 0
        for qi,query in enumerate(query_class_tran_seqs):
            # 返回的总分数最大的一个即为预测结果的类别
            predict = np.argmax(scoreSeqOnTranMats(query, class_tran_mats))
            acc_count += (predict == (qi//qk))

        epoch_acc = acc_count / (qk*n)
        if verbose:
            print("Acc:", epoch_acc)
        tm.step(prt=verbose)
        acc_hist.append(epoch_acc)

    if verbose:
        print("\n")
        print("*"*50)
        print("Avg acc:", sum(acc_hist)/epoch)
        print("95%% belief interval:", calBeliefeInterval(acc_hist))
        print("Total consuming time:", tm.step(prt=False,end=True))

    return sum(acc_hist)/epoch


def gridSearch(c_values, k_values, per_epoch=200):     # 网格搜索聚类类簇数量和截断长度
    re = {}
    for ci,c_num in enumerate(c_values):
        re[c_num] = {}
        for ki,k_num in enumerate(k_values):
            print(ci*len(k_values)+ki+1, "/", len(c_values)*len(k_values))
            mng = PathManager("virushare-20-original")
            # findOptK(mng.WordEmbedMatrix(), k_range=(2,100))
            apiCluster(mng.WordEmbedMatrix(), mng.DataRoot() + "MarkovClusterMapping.json", cluster_num=c_num)
            makeClusteredData(json_path=mng.Folder(),
                              cluster_path=mng.DataRoot() + "MarkovClusterMapping.json",
                              word_map_path=mng.WordIndexMap(),
                              dump_path=mng.DataRoot() + "MarkovClusteredData.npy",
                              max_len=k_num)
            a = scoreMarkovEpisode(clustered_data_path=mng.DataRoot() + "MarkovClusteredData.npy",
                                   epoch=per_epoch,
                                   n_cluster=c_num,
                                   maxlen=k_num,
                                   verbose=False)
            re[c_num][k_num] = a

    return re


def extractBestParam(re):
    best_c = None
    best_k = None
    best_acc = -1

    for ck, cv in re.items():
        for kk, kv in cv.items():
            if kv > best_acc:
                best_acc = kv
                best_c = ck
                best_k = kk

    return best_c, best_k


if __name__ == "__main__":
    epoch = 5000
    seq_len = 50
    n_cluster = 30
    n_range = (15,30)
    mng = PathManager("HKS-api")

    # # # findOptK(mng.WordEmbedMatrix(), k_range=(2,100))
    # apiCluster(mng.WordEmbedMatrix(), mng.DataRoot()+"MarkovClusterMapping.json", cluster_num=n_cluster)
    # makeClusteredData(json_path=mng.Folder(),
    #                   cluster_path=mng.DataRoot()+"MarkovClusterMapping.json",
    #                   word_map_path=mng.WordIndexMap(),
    #                   dump_path=mng.DataRozot()+"MarkovClusteredData.npy",
    #                   max_len=seq_len)
    # scoreMarkovEpisode(clustered_data_path=mng.DataRoot()+"MarkovClusteredData.npy",
    #                    epoch=2000,
    #                    n_cluster=n_cluster,
    #                    maxlen=seq_len)

    # re = gridSearch(c_values=list(range(*n_range)),
    #                 k_values=[i*50 for i in range(1,11)],
    #                 per_epoch=1000)
    # dumpJson(re, mng.DataRoot()+"GSs/GridSearchResult-%dshot-%dway-virushare20.json"%(k,n))
    # re = loadJson(mng.DataRoot()+"GSs/GridSearchResult-%dshot-%dway-virushare20.json"%(k,n))
    # n_cluster, seq_len = extractBestParam(re)
    # n_cluster = int(n_cluster)
    # seq_len = int(seq_len)

    apiCluster(mng.WordEmbedMatrix(), mng.DataRoot()+"MarkovClusterMapping.json", cluster_num=n_cluster)
    makeClusteredData(json_path=mng.Folder(),
                      cluster_path=mng.DataRoot()+"MarkovClusterMapping.json",
                      word_map_path=mng.WordIndexMap(),
                      dump_path=mng.DataRoot()+"MarkovClusteredData.npy",
                      max_len=seq_len)
    scoreMarkovEpisode(clustered_data_path=mng.DataRoot()+"MarkovClusteredData.npy",
                       epoch=epoch,
                       n_cluster=n_cluster,
                       maxlen=seq_len)

