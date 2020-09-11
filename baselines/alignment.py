import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm
import swalign

from utils.color import getRandomColor
from utils.manager import PathManager
from utils.file import dumpJson, loadJson, dumpIterable
from scripts.embedding import aggregateApiSequences
from utils.magic import magicSeed, sample, nRandom
from utils.timer import StepTimer
from utils.stat import calBeliefeInterval

k = 10
qk = 5
n = 10
N = 20

def apiCluster(dict_path, map_dump_path, cluster_num=26):
    api_mat = np.load(dict_path, allow_pickle=True)

    # pca = TSNE(n_components=2)
    # de_api_mat = pca.fit_transform(api_mat)
    # colors = getRandomColor(26, more=False)

    print("Clustering...")
    km = KMeans(n_clusters=cluster_num).fit(api_mat)
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
def convertApiCategory(clst_path, word_map_path, json_path, str_dump_path, max_len=300):
    word_map = loadJson(word_map_path)
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

def align(s1, s2, out):
    match = 2
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)

    # This sets up the aligner object. You must set your scoring matrix, but
    # you can also choose gap penalties, etc...
    sw = swalign.LocalAlignment(scoring)

    # Using your aligner object, calculate the alignment between
    # ref (first) and query (second)
    alignment = sw.align(s1, s2)

    return alignment.identity


def scoreEpisodeAlignment(str_path, epoch=1000, log_path=None, verbose=False,
                          acc_dump_path=None):
    if acc_dump_path is not None:
        if not os.path.exists(acc_dump_path):
            dumpIterable([], "acc", acc_dump_path)
        acc_sum = loadJson(acc_dump_path)['acc']
    else:
        acc_sum = []
    matrix = loadJson(str_path)['strings']
    class_pool = list(range(len(matrix) // N))
    item_pool = set(range(N))
    out = sys.stdout if log_path is None else open(log_path, "w")
    tm = StepTimer(epoch)

    tm.begin()
    for i in range(epoch):
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

        correct_count = 0

        for qi,query in enumerate(queries):
            scores = []
            for si,support in enumerate(supports):
                if verbose:
                    print(qi*n*k+si, "/", n*qk*k*n)
                scores.append(align(support, query, out))

            scores = np.array(scores).reshape(n,k).sum(-1)
            predict = np.argmax(scores)
            correct_count += (predict==(qi//qk))

        epoch_acc = correct_count / (n*qk)
        acc_sum.append(epoch_acc)
        if acc_dump_path is not None:
            dumpIterable(acc_sum, "acc", acc_dump_path)

        print("acc=", epoch_acc)
        tm.step()

    print("\n*********************************************")
    print("Avg acc: ", sum(acc_sum)/epoch)
    print("Total time:", tm.step(prt=False,end=True))
    print("95%% belief interval:", calBeliefeInterval(acc_sum))

    if log_path is not None:
        out.close()








# def genFamilyProtoByMSA(str_path, work_space, proto_dump_path):
#     protos = {}
#     strs = loadJson(str_path)['strings']
#
#     for i in range(0,len(strs)-1,N):
#         print(i,"->",i+N)
#         fa_strs = strs[i:i+N]
#         with open(work_space+"family_"+str(i//N+1)+"_input.txt", "w") as f:
#             try:
#                 for j in range(N):
#                     f.write(f"> {j+1}\n")
#                     f.write(fa_strs[j]+"\n")
#             except Exception as e:
#                 print(f"len={len(fa_strs)} i={i} j={j} msg={str(e)}")
#                 raise RuntimeError





if __name__ == '__main__':
    mng = PathManager("virushare-20-original")
    # apiCluster(mng.WordEmbedMatrix(), mng.DataRoot()+"CategoryMapping.json")
    # convertApiCategory(clst_path=mng.DataRoot()+"CategoryMapping.json",
    #                    word_map_path=mng.WordIndexMap(),
    #                    json_path=mng.DatasetBase()+'all-rmsub/',
    #                    str_dump_path=mng.DataRoot()+"CategorizedStringData(rmsub).json")
    # genFamilyProtoByMSA(str_path=mng.DataRoot()+"CategorizedStringData.json",
    #                     work_space="D:/datasets/virushare-20-original/data/family_protos/",
    #                     proto_dump_path=mng.DataRoot()+"FamilyProtos.txt")
    scoreEpisodeAlignment(str_path=mng.DataRoot()+"CategorizedStringData(rmsub).json",
                          epoch=300,
                          log_path=mng.DataRoot()+'logs/runlog.txt',
                          acc_dump_path=mng.DataRoot()+"logs/Align-Virushare20-%dshot-%dway.json"%(k,n))