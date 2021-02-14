import torch as t
import torch.nn as nn
import numpy as np
import random as rd
from torch.utils.data import DataLoader
from torch.autograd import no_grad
import os
import sys
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt

from components.datasets import SeqFileDataset
from models.SIMPLE import SIMPLE
from utils.manager import PathManager, TrainingConfigManager
from utils.file import loadJson
from components.task import ImpEpisodeTask

def drawHeatmap(data, path, title, col_labels, row_labels, cbar_label, formatter="%s", **kwargs):
    mpl.rcParams['axes.linewidth'] = 0.1
    mpl.rcParams['xtick.minor.size'] = 0.1
    mpl.rcParams['xtick.minor.width'] = 0.1
    mpl.rcParams['ytick.minor.size'] = 0.1
    mpl.rcParams['ytick.minor.width'] = 0.1

    fig, ax = plt.subplots(figsize=(12, 12), dpi=600)  #
    im = ax.imshow(data, **kwargs)
    for y in range(20):
        ax.axhline(0.5 + y, linestyle='-', color='k', lw=0.5)
    ax.axhline(-0.5, linestyle='-', color='k', lw=0.8)
    ax.axhline(-0.5 + 20, linestyle='-', color='k', lw=0.8)
    for x in range(20):
        ax.axvline(0.5 + x, linestyle='-', color='k', lw=0.5)
    ax.axvline(-0.5, linestyle='-', color='k', lw=0.8)
    ax.axvline(-0.5 + 20, linestyle='-', color='k', lw=0.8)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontsize(10)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    # ... and label them with the respective list entries
    ax.tick_params(width=0.1, length=0.1)
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", verticalalignment='top')

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if data[i][j] >= 0.01:
                color = 'w' if i==j else 'k'
                text = ax.text(j, i, formatter % data[i][j],
                               ha="center", va="center", color=color, fontsize=9)

    ax.set_title(title)
    fig.tight_layout()

    if path is None:
        plt.show()
    else:
        fig.savefig(path, format='png', dpi=600)
    # plt.savefig()


# cfg = TrainingConfigManager('../run/testConfig.json')
# datasetBasePath = cfg.systemParams()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.deviceId())
# data_folder = cfg.dataset()#'virushare_20_image'
#
# k,n,qk,N = cfg.taskParams()
# model_type, model_name = cfg.model()
#
# version = cfg.version()
#
# TestingEpoch = cfg.epoch()
# USED_SUB_DATASET = cfg.subDataset()
# MODEL_RANDOM_STATE = cfg.isRandom()
#
# test_path_manager = PathManager(dataset=data_folder,
#                                d_type=USED_SUB_DATASET,
#                                model_name=model_name,
#                                version=version)
#
# model_cfg = TrainingConfigManager(test_path_manager.Doc()+'config.json')
# modelParams = model_cfg.modelParams()
#
# LRDecayIters, LRDecayGamma, optimizer_type,\
# weight_decay, loss_func, default_lr, lrs, \
# taskBatchSize, criteria = model_cfg.trainingParams()
#
# test_dataset = SeqFileDataset(test_path_manager.FileData(),
#                                test_path_manager.FileSeqLen(),
#                                N)
#
# expand = True if loss_func=='mse' else False
# test_task = ImpEpisodeTask(k, qk, n, N, test_dataset,
#                                   cuda=True, expand=expand)
#
# state_dict = t.load(test_path_manager.Model(type=cfg.loadBest()))
# word_matrix = state_dict['Embedding.weight']
# # loss = t.nn.NLLLoss().cuda() if loss_func=='nll' else t.nn.MSELoss().cuda()
#
# model = SIMPLE(pretrained_matrix=word_matrix, **modelParams)
# model.load_state_dict(state_dict)
# model = model.cuda()
#
# model.eval()
#
# label_list = []
# predict_list = []
# for epoch in range(TestingEpoch):
#     if epoch % 100 == 0:
#         print(epoch,"/",TestingEpoch)
#     supports, queries, sup_len, que_len, sup_labels, que_labels = test_task.episode()
#
#     pred_labels, loss_val = model(supports, queries, sup_len, que_len, sup_labels, que_labels)
#
#     pred_labels, real_labels = test_task.restoreLabels(pred_labels.cpu())
#     label_list += pred_labels.tolist()
#     predict_list += real_labels.tolist()
#
# acc = (t.LongTensor(label_list)==t.LongTensor(predict_list)).sum().item() / len(label_list)
# cfm = confusion_matrix(label_list, predict_list)
#
# np.save(test_path_manager.DatasetBase()+"data/cfm_data.npy",
#         cfm)

########################################################
path = 'C:/Users/Asichurter/Desktop/fsdownload/cfm_data.npy'
idx_path = 'C:/Users/Asichurter/Desktop/fsdownload/idxMapping.json'

cfm = np.load(path)
idx_mapping = loadJson(idx_path)

row_sum = cfm.sum(1)[:,None].repeat(len(cfm),1)
normalized_cfm = cfm / row_sum


ticks = [idx_mapping[str(i)] for i in range(20)]
drawHeatmap(normalized_cfm,
            'C:/Users/Asichurter/Desktop/fsdownload/cfm.png',
            '', ticks, ticks, "acc", formatter="%.3f", cmap="GnBu")

# from utils.manager import PathManager
# from utils.file import loadJson
# p = PathManager('HKS')
# pt = p.DataRoot()+'train/idxMapping.json'
# js = loadJson(pt)
# s = ''
# for k,v in js.items():
#     s += v
#     s += ','
# print(s)