# -*- coding: utf-8

import numpy as np
import torch as t
import torch.nn.functional as F

from scripts.dataset import makeDataFile, makeDatasetDirStruct, splitDatas, \
                                dumpDatasetSplitStruct, revertDatasetSplit, \
                                deleteDatasetSplit
from utils.file import loadJson, dumpJson
from utils.manager import PathManager
from scripts.reshaping import makeMatrixData
from scripts.preprocessing import apiStat, removeApiRedundance, statSatifiedClasses, \
                                    collectJsonByClass, mappingApiNormalize, filterApiSequence, \
                                    renameItemFolder
from models.ProtoNet import IncepProtoNet
from utils.matrix import batchDot
# from scripts.embedding import *
from extractors.ngram import statNGram, convertToNGramSeq
from extractors.TFIDF import calTFIDF

from config import generateConfigReport

# 生成报告总结
################################################################
# generateConfigReport(dataset='virushare-20-3gram', include_result=True,
#                      dump_path='../docs/virushare-20-3gram-summary.json')
################################################################

# 生成/还原 数据集分割文件
###############################################################
# manager = PathManager(dataset='virushare-20-3gram')
# dumpDatasetSplitStruct(base_path=manager.DatasetBase(),
#                        dump_path=manager.DatasetBase()+'data/split_4.json')
# revertDatasetSplit(dataset='virushare-20-h3gram',
#                    dump_path='/home/asichurter/datasets/JSONs/virushare-20-3gram/data/split_4.json')
###############################################################


# 分割数据集
################################################################
# base = '/home/omnisky/Asichurter/ApiData/LargePE-80/'
# # man = PathManager(dataset='virushare-20-3gram')
# # deleteDatasetSplit(dataset_base=man.DatasetBase())
# splitDatas(src=base+'all/',
#            dest=base+'train/',
#            ratio=-1,
#            mode='c',
#            is_dir=True)
# splitDatas(src=base+'train/',
#            dest=base+'validate/',
#            ratio=15,
#            mode='x',
#            is_dir=True)
# splitDatas(src=base+'train/',
#            dest=base+'test/',
#            ratio=15,
#            mode='x',
#            is_dir=True)
################################################################

# 制作基于下标的数据集
################################################################
# makeDatasetDirStruct(base_path="/home/asichurter/datasets/JSONs/virushare-20-h3gram/")
for d_type in ['train', 'validate', 'test']:
    manager = PathManager(dataset='LargePE-80', d_type=d_type)

    makeDataFile(json_path=manager.Folder(),
                 w2idx_path=manager.WordIndexMap(),
                 seq_length_save_path=manager.FileSeqLen(),
                 data_save_path=manager.FileData(),
                 num_per_class=80,
                 max_seq_len=200)
################################################################

# renameItemFolder('/home/asichurter/datasets/JSONs/LargePE-100-original/')

# 统计序列长度分布
################################################################
# apiStat('/home/omnisky/Asichurter/ApiData/LargePE-100-original/',
#         ratio_stairs=[500, 1000, 2000, 4000, 5000, 10000, 20000, 50000],
#         dump_report_path='/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_report.json',#None,#
#         dump_apiset_path='/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_set.json',#None
#         class_dir=False)
################################################################



# 统计满足数量规模的类别
################################################################
# statSatifiedClasses(pe_path='/home/omnisky/Asichurter/ApiData/LargePE-100-PE-dummy/',
#                     json_path='/home/omnisky/Asichurter/ApiData/LargePE-100-original/',
#                     report_path='/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_report.json',
#                     stat_stairs=[50, 60, 75, 80, 90, 100],
#                     count_dump_path='/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_scale_report.json')
################################################################

# 按照已经知道的满足规模的类进行收集
################################################################
# makeDatasetDirStruct(base_path='/home/omnisky/Asichurter/ApiData/LargePE-80/')
# collectJsonByClass(pe_path='/home/omnisky/Asichurter/ApiData/LargePE-100-PE-dummy/',
#                    json_path='/home/omnisky/Asichurter/ApiData/LargePE-100-original/',
#                    dst_path='/home/omnisky/Asichurter/ApiData/LargePE-80/all/',
#                    report_path='/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_report.json',
#                    num_per_class=80,
#                    selected_classes=["Backdoor.Win32.Ceckno", "Backdoor.Win32.Popwin", "Trojan-Downloader.Win32.Suurch", "Trojan-PSW.Win32.VB", "Backdoor.Win32.Hupigon", "Trojan-Spy.Win32.Iespy", "Backdoor.Win32.Singu", "Trojan-Downloader.Win32.Zanoza", "Backdoor.Win32.BO2K", "Trojan.Win32.Delf", "Backdoor.Win32.Prorat", "Email-Worm.Win32.Warezov", "Trojan-Downloader.Win32.VB", "Virus.Win32.VB", "Virus.Win32.Xorer", "Trojan-PSW.Win32.Delf", "Trojan-Proxy.Win32.Puma", "Backdoor.Win32.Ciadoor", "Trojan-Proxy.Win32.Pixoliz", "Trojan-Downloader.Win32.Agent", "Worm.Win32.VB", "Trojan-PSW.Win32.QQPass", "Backdoor.Win32.Agent", "Backdoor.Win32.Rukap", "Trojan-Spy.Win32.BZub", "Trojan-Dropper.Win32.Delf", "Trojan.Win32.Chinaad", "Backdoor.Win32.Cakl", "Trojan-GameThief.Win32.Nilage", "Trojan-Clicker.Win32.Small", "Trojan.Win32.Buzus", "HackTool.Win32.VB", "Backdoor.Win32.BlackHole", "Trojan.Win32.CDur", "Trojan.Win32.VB", "Trojan.Win32.Obfuscated", "Trojan-Banker.Win32.Banker", "Backdoor.Win32.SdBot", "Trojan-GameThief.Win32.OnLineGames", "Trojan-Banker.Win32.Banbra", "Trojan-Downloader.Win32.Wintrim", "Trojan-GameThief.Win32.Tibia", "Trojan-Spy.Win32.Pophot", "Trojan-Downloader.Win32.Adload", "Trojan-PSW.Win32.Agent", "Trojan-PSW.Win32.OnLineGames", "Trojan-Dropper.Win32.FriJoiner", "Trojan-PSW.Win32.Gamec", "Trojan-Dropper.Win32.VB", "Trojan-Downloader.Win32.Zlob", "Trojan.Win32.Diamin", "Backdoor.Win32.Rbot", "Net-Worm.Win32.Kolabc", "Backdoor.Win32.Bifrose", "Trojan.Win32.BHO", "Trojan-Spy.Win32.Agent", "Trojan.Win32.Slefdel", "Trojan-Downloader.Win32.Swizzor", "Backdoor.Win32.Httpbot", "Trojan-PSW.Win32.Lmir", "Trojan.Win32.Regrun", "Trojan-PSW.Win32.Maran", "Trojan-Spy.Win32.FlyStudio", "Worm.Win32.Downloader", "Trojan.Win32.Monder", "Rootkit.Win32.Vanti", "Backdoor.Win32.Prosti", "Trojan-Clicker.Win32.Delf", "Trojan-Spy.Win32.Delf", "Rootkit.Win32.Podnuha", "Net-Worm.Win32.Kolab", "Worm.Win32.Viking", "Worm.Win32.AutoRun", "Backdoor.Win32.Sinowal", "Trojan.Win32.Monderb", "Trojan-PSW.Win32.Nilage", "Trojan-PSW.Win32.WOW", "Trojan-Proxy.Win32.Agent", "Trojan-PSW.Win32.QQShou", "Trojan-PSW.Win32.QQRob", "Trojan.Win32.Midgare", "Trojan-Downloader.Win32.Winlagons", "Trojan-Spy.Win32.KeyLogger", "Trojan-Downloader.Win32.Small", "Trojan-GameThief.Win32.WOW", "Trojan-Spy.Win32.Flux", "Backdoor.Win32.Frauder", "Trojan-GameThief.Win32.Lmir", "Trojan.Win32.Zapchast", "Trojan-Spy.Win32.Banbra", "Backdoor.Win32.Small", "Trojan-Clicker.Win32.VB", "Trojan.Win32.Vapsup", "Trojan.Win32.KillAV", "Trojan-Spy.Win32.Ayolog", "Backdoor.Win32.Shark", "Worm.Win32.Runfer", "Backdoor.Win32.Delf", "Trojan-Spy.Win32.Banker", "Trojan-Clicker.Win32.Agent", "Trojan.Win32.Qhost"])
################################################################


# 将数据集转化为下标形式来减少内存占用
################################################################
# apiSet = loadJson('/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_set.json')
# apis = apiSet['api_set']
# mapping = {name:str(i) for i,name in enumerate(apis)}
# apiSet['api_map'] = mapping
# mappingApiNormalize(json_path='/home/omnisky/Asichurter/ApiData/LargePE-80/all/',
#                     mapping=mapping,
#                     is_class_dir=True)
# # save back the api mapping
# dumpJson(apiSet, '/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_set.json')
################################################################


# 训练W2V模型(GloVe)
################################################################
# manager = PathManager(dataset='virushare-10-seq', d_type='all')
# seqs = aggregateApiSequences(manager.Folder())
# trainW2Vmodel(seqs,
#               save_matrix_path=manager.WordEmbedMatrix(),
#               save_word2index_path=manager.WordIndexMap(),
#               size=300)
################################################################

# 统计ngram
################################################################
# print('Removing Redundance...')
# removeApiRedundance(json_path='/home/omnisky/Asichurter/ApiData/LargePE-100-original/',
#                     class_dir=False)
#
# # man = PathManager(dataset='virushare-20-h3gram', d_type='all')
# print('Stating NGram...')
# ngram_dict = statNGram(parent_path='/home/omnisky/Asichurter/ApiData/LargePE-100-original/',
#                        window=3,
#                        dict_save_path='/home/omnisky/Asichurter/report/LargePE-100_3gram_api_set.json',
#                        frequency_stairs=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
#                        class_dir=False)
# #
# # num = int(input('NGram >> '))
# # #
# # d = loadJson('/home/asichurter/datasets/reports/virushare-20_h3gram_api_freq.json')
# #
# print('Converting NGram...')
# convertToNGramSeq(parent_path='/home/omnisky/Asichurter/ApiData/LargePE-100-original/',
#                   window=3,
#                   ngram_dict=ngram_dict,
#                   ngram_max_num=None,
#                   class_dir=False)
################################################################



# 计算tdidf并且根据该值过滤API
 #################################################################
# api_set = loadJson('/home/omnisky/Asichurter/report/LargePE-100_3gram_api_set.json')
# dict_map = {k:i for i,k in enumerate(api_set)}
# dumpJson(dict_map, '/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_dictmap.json')
# top_k_apis = calTFIDF(dataset_path='/home/omnisky/Asichurter/ApiData/LargePE-100-original/',
#                       dict_map_path='/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_dictmap.json',
#                       is_class_dir=False,
#                       tfidf_dump_path='/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_val.json',
#                       top_k=2000)
# api_tfidf = loadJson('/home/omnisky/Asichurter/report/LargePE-100_3gram_tfidf_api_val.json')
# print('Sorting...')
# api_tfidf = sorted(api_tfidf.items(), key=lambda item:item[1], reverse=True)
# api_list = [api[0] for i,api in enumerate(api_tfidf) if i < 2000]
# print('Filtering...')
# filterApiSequence(json_path='/home/omnisky/Asichurter/ApiData/LargePE-100-original/',
#                   api_list=api_list,
#                   keep_or_filter=False)
#################################################################

# 转化数据集
################################################################
# manager = PathManager(dataset='virushare_20', d_type='train')
# matrix = np.load(manager.WordEmbedMatrix(), allow_pickle=True)
# mapping = get1DRepreByRuduc(matrix)
# seq = t.load(manager.FileData())
# seq = reshapeSeqToMatrixSeq(seq, mapping, flip=True)
# t.save(t.Tensor(seq), manager.FileData())
# makeMatrixData(dataset='virushare_20', shape=(40, 10, 10))

# 翻转序列
################################################################
# a[:,1::2] = np.apply_along_axis(lambda x: np.flipud(x), 2, a[:,1::2])
################################################################


# 测试Inception模型
################################################################
# m = IncepProtoNet(channels=[1,32,1],
#                   depth=3)
# s = t.randn((5, 5, 100, 8, 5))
# q = t.randn((75, 100, 8, 5))
# out = m(s, q)
################################################################

# 测试自注意力
################################################################
# x = t.Tensor([
#     [
#         [1,2,3],
#         [3,4,5]
#     ],
#     [
#         [5, 6, 7],
#         [7, 8, 9]
#     ]
# ])
#
# Q = t.nn.Linear(3,3, bias=False)
# K = t.nn.Linear(3,3, bias=False)
# V = t.nn.Linear(3,3, bias=False)
#
# t.nn.init.constant_(Q.weight, 1)
# t.nn.init.constant_(K.weight, 1)
# t.nn.init.constant_(V.weight, 1)
#
# k = K(x)
# q = Q(x)
# v = V(x)
#
# w = batchDot(q, k, transpose=True)
# p = t.softmax(w, dim=2)
# z = batchDot(p, v, transpose=False)
################################################################

# API名称规范化
################################################################
# mappingApiNormalize(path,
#                     mapping={
#                         "RegCreateKeyExA": "RegCreateKey",
#                         "RegCreateKeyExW": "RegCreateKey",
#                         "RegDeleteKeyA": "RegDeleteKey",
#                         "RegDeleteKeyW": "RegDeleteKey",
#                         "RegSetValueExA": "RegSetValue",
#                         "RegSetValueExW": "RegSetValue",
#                         "RegDeleteValueW": "RegDeleteValue",
#                         "RegDeleteValueA": "RegDeleteValue",
#                         "RegEnumValueW": "RegEnumValue",
#                         "RegEnumValueA": "RegEnumValue",
#                         "RegQueryValueExW": "RegQueryValue",
#                         "RegQueryValueExA": "RegQueryValue",
#                         "CreateProcessInternalW": "CreateProcess",
#                         "NtCreateThreadEx": "NtCreateThread",
#                         "CreateRemoteThread": "CreateRemoteThread",
#                         "CreateThread": "CreateThread",
#                         "NtTerminateProcess": "TerminateProcess",
#                         "NtOpenProcess": "OpenProcess",
#                         "InternetOpenUrlA": "InternetOpenUrl",
#                         "InternetOpenUrlW": "InternetOpenUrl",
#                         "InternetOpenW": "InternetOpen",
#                         "InternetOpenA": "InternetOpen",
#                         "InternetConnectW": "InternetConnect",
#                         "InternetConnectA": "InternetConnect",
#                         "HttpOpenRequestW": "HttpOpenRequest",
#                         "HttpOpenRequestA": "HttpOpenRequest",
#                         "HttpSendRequestA": "HttpSendRequest",
#                         "HttpSendRequestW": "HttpSendRequest",
#                         "ShellExecuteExW": "ShellExecute",
#                         "LdrLoadDll": "LdrLoadDll",
#                         "CopyFileW": "CopyFile",
#                         "CopyFileA": "CopyFile",
#                         "CopyFileExW": "CopyFile",
#                         "NtCreateFile": "CreateFile",
#                         "DeleteFileW": "DeleteFile",
#                         "NtDeleteFile": "NtDeleteFile",
#                     })
################################################################
