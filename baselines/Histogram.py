'''
    This script is used to run few-shot classification on frequency
    histogram data. Operations of making datasets and doing classification
    are all included in this file.
'''



import os
import torch as t
import numpy as np
from tqdm import tqdm
import random as rd
import sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils.file import loadJson, dumpJson
from utils.magic import magicSeed, magic
from utils.stat import calBeliefeInterval
from utils.manager import PathManager



##############################################
# 根据序列数据集(ngram,api)，获取序列元素的频率直方图
# 需要提供所有序列元素的实值映射(data_dict_path)
##############################################
def getHist(src_path, dst_path,
            dict_map_path,          # 将序列元素转化为一个实值的映射的路径，通常为wordMap.json
            is_class_dir=True):

    os.system(f'rm -rf {dst_path}*')    # 删除目标文件夹中的所有元素

    value_map = loadJson(dict_map_path)
    value_min = min(value_map.values())
    value_max = max(value_map.values())
    value_size = value_max - value_min + 1

    for folder in tqdm(os.listdir(src_path)):

        if is_class_dir:
            items = os.listdir(src_path+folder+'/')
            if os.path.exists(dst_path+folder+'/'):
                raise RuntimeError("目标文件夹%s已存在！"%folder)
            else:
                os.mkdir(dst_path+folder+'/')

        else:
            items = [folder+'.json']

        for item in items:
            data = loadJson(src_path+folder+'/'+item)
            seqs = data['apis']

            #　映射token为实值
            seqs = [value_map[s] for s in seqs]

            hist, bins_sep = np.histogram(seqs,
                                        range=(value_min-0.5,value_max+0.5),   # 这样划分使得每一个int值都处于一个单独的bin中
                                        bins=value_size,
                                        normed=True)

            hist_warpper = {"hist":hist.tolist()}
            dumpJson(hist_warpper, dst_path+folder+'/'+item)

    print("- Done -")


########################################################
# 收集位于json中的直方图数据，整合成为一个numpy矩阵，其中矩阵的
# 形状为:
# [ 类别， 样本， 维度]
########################################################
def makeClasswiseHistDataset(json_folder_path, dst_path):
    # shape: [class, num, dim]
    data_matrix = []

    for folder in tqdm(os.listdir(json_folder_path)):
        folder_path = json_folder_path+folder+'/'
        class_matrix = []

        for item in os.listdir(folder_path):
            item_path = folder_path + item
            hist = loadJson(item_path)['hist']
            class_matrix.append(hist)

        data_matrix.append(class_matrix)

    data_matrix = np.array(data_matrix)
    np.save(dst_path, data_matrix)

    print('- Done -')


#############################################################
# 生成分类任务的标签空间，并将训练数据矩阵按照 训练/测试 对每个
# 类内样本进行划分
#############################################################
def splitMatrixForTrainTest(matrix,
                            class_num,              # 类数量
                            testnum_per_class,      # 测试集中每个类的样本数量
                            ravel=True,             # 是否展平标签
                            task_seed=None,
                            sampling_seed=None):

    assert test_num_per_class < matrix.shape[1], "每个类的测试样本数量必须少于总数量!"

    task_seed = rd.randint(0,magic) if task_seed is None else task_seed
    sampling_seed = rd.randint(0,magic) if sampling_seed is None else sampling_seed

    total_class_num = matrix.shape[0]
    instance_num = matrix.shape[1]
    dim = matrix.shape[2]

    # 选中n-way任务中的类
    rd.seed(task_seed)
    selected_class_idxes = rd.sample(list(range(total_class_num)), class_num)
    matrix = matrix[selected_class_idxes]

    idxes_full_set = set(range(0,instance_num))
    rd.seed(sampling_seed)
    class_seeds = [rd.randint(0,magic) for i in range(class_num)]

    class_wise_test_idxes = []
    class_wise_train_idxes = []
    for class_seed in class_seeds:
        rd.seed(class_seed)

        test_idxes = rd.sample(idxes_full_set, testnum_per_class)
        # 训练集下标取所有测试集下标的补集
        train_idxes = list(idxes_full_set.difference(set(test_idxes)))

        # 由于要选择第1维度上的样本，第2维度上的数据的所有选择值都需要被置为下标
        class_wise_test_idxes.append([[idx]*dim for idx in test_idxes])
        class_wise_train_idxes.append([[idx]*dim for idx in train_idxes])

    torch_matrix = t.Tensor(matrix)
    torch_train_idxes = t.LongTensor(class_wise_train_idxes)
    torch_test_idxes = t.LongTensor(class_wise_test_idxes)

    # 利用torch.gather进行下标选择
    torch_train_matrix = t.gather(torch_matrix, 1, torch_train_idxes)
    torch_test_matrix = t.gather(torch_matrix, 1, torch_test_idxes)

    if ravel:
        torch_train_matrix = torch_train_matrix.view((class_num*train_num_per_class, -1))
        torch_test_matrix = torch_test_matrix.view((class_num*test_num_per_class, -1))

    # 最终返回分割后的训练集，测试集和全部数据
    return torch_train_matrix.numpy(), torch_test_matrix.numpy()#, torch_matrix.numpy()


def reduceMatrixDim(matrix, n_comp):
    output_shape = (matrix.shape[0], matrix.shape[1], -1)
    matrix = matrix.reshape((matrix.shape[0]*matrix.shape[1], -1))

    reduction = PCA(n_components=n_comp)

    reduced_matrix = reduction.fit_transform(matrix)

    return reduced_matrix.reshape(output_shape)

##############################################################
# 根据提供的类数量和样本数量，生成样本的标签
##############################################################
def makeLabels(class_num, instance_num):
    # 创建不同类的连续int标签
    labels = np.linspace(0,class_num-1,class_num).astype(np.int)
    # 每个类标签重复样本数量次
    labels = np.expand_dims(labels,axis=-1).repeat(instance_num,axis=-1)
    # 展平标签
    labels = np.ravel(labels)
    return labels


##############################################################
# 利用分类器完成分类任务
##############################################################
def doClassification(train_data, train_label, test_data, test_label,
                     method='knn',
                     **params):
    if method == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    elif method == 'svm':
        classifier = SVC()
    else:
        raise ValueError("不支持的分类器:"+str(method))

    classifier.fit(train_data, train_label)

    predicts = classifier.predict(test_data)
    acc = (predicts==test_label).sum() / test_data.shape[0]

    return acc





if __name__ == '__main__':
    src_dataset_name = "virushare-45"
    sub_dataset = 'all'
    dst_dataset_name = "virushare-45-hist"
    path_manager = PathManager('')
    parent_path = path_manager.ParentPath

    assert src_dataset_name != dst_dataset_name

    #-----------------------------------------------------------------------------
    getHist(src_path=parent_path + src_dataset_name +'/' + sub_dataset + '/',  #'/home/asichurter/datasets/JSONs/virushare-45-rmsub/test/',
            dst_path=parent_path + dst_dataset_name +'/' + sub_dataset + '/',  #'/home/asichurter/datasets/JSONs/virushare-45-rmsub-hist/test/',
            dict_map_path=parent_path+src_dataset_name+'/data/wordMap.json',
            is_class_dir=True)
    makeClasswiseHistDataset(json_folder_path=parent_path + dst_dataset_name +'/' + sub_dataset + '/',
                             dst_path=parent_path + dst_dataset_name +'/data-%s.npy' % sub_dataset)
    #-----------------------------------------------------------------------------


    iteration = 1000            # 测试轮数
    n = 5                       # 分类类数量
    test_num_per_class = 40     # 每一个类中测试样本的数量
    reduction_n_comp = 0.9      # 保留0.9方差
    knn_k = 1                   # kNN的近邻数量

    print('Loading data...')
    mat = np.load(parent_path + dst_dataset_name +'/data-%s.npy' % sub_dataset)
    mat = reduceMatrixDim(mat, reduction_n_comp)

    train_num_per_class = mat.shape[1] - test_num_per_class

    total_acc = 0.
    hist_acc = []

    for i in range(iteration):
        print('iter %d,'%i, end=' ')
        train_data, test_data = splitMatrixForTrainTest(mat,
                                                        class_num=n,
                                                        testnum_per_class=test_num_per_class,
                                                        ravel=True)

        train_label = makeLabels(class_num=n, instance_num=train_num_per_class)
        test_label = makeLabels(class_num=n, instance_num=test_num_per_class)

        acc = doClassification(train_data, train_label,
                               test_data, test_label,
                               # method='svm')
                               method='knn', n_neighbors=knn_k)

        hist_acc.append(acc)

        print('acc= %f'%acc)

    print('\nAverage Acc: %f±%.5f'%(np.mean(hist_acc),calBeliefeInterval(hist_acc)))