import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import torch as t
from torch.utils.data.dataloader import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,MDS,LocallyLinearEmbedding
import numpy as np

from models.ProtoNet import ProtoNet
from utils.color import getRandomColor
from models.IMP import IMP
from models.SIMPLE import SIMPLE
from models.HybridIMP import HybridIMP
from models.NnNet import NnNet

from components.datasets import SeqFileDataset
from utils.manager import PathManager, TrainingConfigManager
from utils.training import batchSequenceWithoutPad
from components.task import ImpEpisodeTask, ProtoEpisodeTask
from utils.magic import magicSeed

# ***********************************************************
data_dataset_name = "HKS"
model_dataset_name = "HKS"
dataset_subtype = 'test'

model_name = 'ProtoNet'

version = 308
N = 20
plot_option = 'entire'#'entire'
k, n, qk = 10, 5, 10
figsize = (6,6)

task_seed = magicSeed()#4160148##4488524#4524396
sampling_seed = 5791326#magicSeed()#4160164##4488540#4524414   # SIMPLE seed: 5331044

axis_off = True
plot_support_independently = False
max_plot_class = 20
selected_idxes = [6,8,10,16,18]
reducer = 'tsne'
# ***************************************************************************


data_path_man = PathManager(dataset=data_dataset_name,
                           d_type=dataset_subtype)
model_path_man = PathManager(dataset=model_dataset_name,
                             version=version,
                             model_name=model_name)

################################################
#----------------------读取模型参数------------------
################################################

model_cfg = TrainingConfigManager(model_path_man.Doc()+'config.json')

modelParams = model_cfg.modelParams()

dataset = SeqFileDataset(data_path_man.FileData(), data_path_man.FileSeqLen(), N=N)

state_dict = t.load(model_path_man.Model() + '_v%s.0' % version)
# state_dict = t.load(path_man.DatasetBase()+'models/ProtoNet_v105.0')
word_matrix = state_dict['Embedding.weight']

if model_name == 'IMP':
    model = IMP(word_matrix,
                **modelParams)
elif model_name == 'SIMPLE':
    model = SIMPLE(word_matrix,
                   **modelParams)
elif model_name == 'HybridIMP':
    model = HybridIMP(word_matrix,
                      **modelParams)
elif model_name == 'ProtoNet':
    model = ProtoNet(word_matrix,
                     **modelParams)
elif model_name == 'NnNet':
    model = NnNet(word_matrix, **modelParams)

model.load_state_dict(state_dict)
model = model.cuda()
model.eval()

print('task_seed:', task_seed)
print('sampling_seed', sampling_seed)

if plot_option == 'entire':
    dataloader = DataLoader(dataset, batch_size=N, collate_fn=batchSequenceWithoutPad)

    datas = []
    original_input = []

    if reducer == 'pca':
        reduction = PCA(n_components=2)
    elif reducer == 'tsne':
        reduction = TSNE(n_components=2, random_state=int(sampling_seed))
    else:
        raise ValueError

    class_count = 0

    for i,(x,_,lens) in enumerate(dataloader):
        class_count += 1
        original_input.append(x.tolist())
        x = x.cuda()
        x = model._embed(x, lens).cpu().view(N,-1).detach().numpy()
        datas.append(x)

        if i+1 == max_plot_class:
            break

    # datas = np.array(datas).reshape((class_count*N,-1))
    if selected_idxes is None:
        selected_idxes = list(range(class_count))

    datas = np.array(datas).reshape((class_count,N,-1))[selected_idxes].reshape(len(selected_idxes)*N,-1)
    datas = reduction.fit_transform(datas)

    colors = ['darkgreen',
 'purple',
 'orange',
 'steelblue',
 'red',
 'teal',
 'yellowgreen',
 'gold',
 'magenta',
 'deepskyblue',
 'blueviolet',
 'red',
 'black',
 'bisque',
 'violet',
 'hotpink',
 'firebrick',
 'darkseagreen',
 'pink',
 'lime']#getRandomColor(class_count)


    datas = datas.reshape((len(selected_idxes),N,2))
    # datas = datas.reshape((class_count,N,2))

    if class_count > len(colors):
        colors = getRandomColor(class_count, more=True)

    plt.figure(figsize=figsize, dpi=300)
    if axis_off:
        plt.axis('off')
    for i in range(len(datas)):
        plt.scatter(datas[i,:,0],datas[i,:,1],color=colors[i],marker='o',label=i)

    # plt.legend()
    # plt.show()

    original_input = np.array(original_input)

# ***************************************************************************

elif plot_option == 'episode':
    if model_name in ['IMP', 'SIMPLE']:
        task = ImpEpisodeTask(k,qk,n,N,
                              dataset,expand=False)
        support_, query_, *others = task.episode(task_seed=task_seed,
                                                 sampling_seed=sampling_seed)
        support, query, acc = model(support_, query_, *others, if_cache_data=True)
        clusters, cluster_labels = model.Clusters.squeeze().cpu().detach(), \
                                   model.ClusterLabels.squeeze().cpu().detach().numpy()
    else:
        task = ProtoEpisodeTask(k, qk, n, N, dataset, expand=False)
        (support_, query_, *lens), labels = task.episode(task_seed=task_seed, sampling_seed=sampling_seed)
        support, query, clusters, sims = model(support_, query_, *lens, return_embeddings=True)
        clusters = clusters.cpu().detach()
        cluster_labels = np.arange(0, n)
        pred_labels = t.argmax(sims, dim=1)
        acc = (pred_labels==labels).sum().item() / len(labels)


    if reducer == 'pca':
        reduction = PCA(n_components=2)
    elif reducer == 'tsne':
        reduction = TSNE(n_components=2)
    else:
        raise ValueError

    support = support.cpu().detach().view(k*n, -1)
    query = query.cpu().detach().view(qk*n,-1)
    union = t.cat((support,query,clusters),dim=0).numpy()

    union = reduction.fit_transform(union)
    support = union[:n*k].reshape((n,k,2))
    query = union[n*k:-len(clusters)].reshape((n,qk,2))
    clusters = union[-len(clusters):]
    query_ = query_.cpu().numpy().reshape((n,qk,-1))

    colors = ['darkgreen',
 'purple',
 'orange',
 'steelblue',
 'red',
 'teal',
 'yellowgreen',
 'gold',
 'magenta',
 'deepskyblue',
 'blueviolet',
 'red',
 'black',
 'bisque',
 'violet',
 'hotpink',
 'firebrick',
 'darkseagreen',
 'pink',
 'lime']#getRandomColor(n)

    # 不带测试数据
    if plot_support_independently:
        plt.figure(figsize=figsize)
        plt.title(f'cluster={len(clusters)}({acc})')
        if axis_off:
            plt.axis('off')
        for i in range(n):
            plt.scatter(support[i,:,0],support[i,:,1],color=colors[i],marker='o')
            # plt.scatter(query[i,:,0],query[i,:,1],color=colors[i],marker='*')

            class_clusters = clusters[cluster_labels==i]
            plt.scatter(class_clusters[:,0],class_clusters[:,1],color=colors[i],marker='x',edgecolors='k',s=80)
            plt.scatter(class_clusters[:,0],class_clusters[:,1],marker='o',c='',edgecolors='k',s=80)

        plt.show()

    # 带测试数据
    plt.figure(figsize=figsize)
    plt.title(f'cluster={len(clusters)}({acc})')
    if axis_off:
        plt.axis('off')
    for i in range(n):
        plt.scatter(support[i,:,0],support[i,:,1],color=colors[i],marker='o',label=i)
        plt.scatter(query[i,:,0],query[i,:,1],color=colors[i],marker='*')

        class_clusters = clusters[cluster_labels==i]
        plt.scatter(class_clusters[:,0],class_clusters[:,1],color=colors[i],marker='x',edgecolors='k',s=80)
        plt.scatter(class_clusters[:,0],class_clusters[:,1],marker='o',c='',edgecolors='k',s=80)

    print('acc: ', acc)
    plt.legend()
    plt.show()

    







