# -*- coding: utf-8

import numpy as np
import torch as t
import torch.nn.functional as F
import os

from preliminaries.dataset import makeDataFile, makeDatasetDirStruct, splitDatas, \
    dumpDatasetSplitStruct, revertDatasetSplit, \
    deleteDatasetSplit, renameItemFolder
from preliminaries.embedding import aggregateApiSequences
from utils.file import loadJson, dumpJson
from utils.manager import PathManager
from preliminaries.reshaping import makeMatrixData
from preliminaries.preprocessing import apiStat, removeApiRedundance, statSatifiedClasses, \
                                    collectJsonByClass, mappingApiNormalize, filterApiSequence, \
                                    collectOriginalHKS
from models.ProtoNet import IncepProtoNet
# from preliminaries.embedding import *
from extractors.ngram import statNGram, convertToNGramSeq
from extractors.TFIDF import calTFIDF

from config import generateConfigReport

# makeDatasetDirStruct(base_path="D:/datasets/HKS-api/")
# collectOriginalHKS(ori_path='D:/datasets/HKS-original/api/',
#                    existed_dataset_path="D:/datasets/HKS/all/",
#                    dump_path="D:/datasets/HKS-api/all/")

# 生成报告总结
################################################################
# generateConfigReport(dataset='virushare-20-3gram', include_result=True,
#                      dump_path='../docs/virushare-20-3gram-summary.json')
################################################################

# 生成/还原 数据集分割文件
###############################################################
# manager = PathManager(dataset='LargePE-Per35')
# dumpDatasetSplitStruct(base_path=manager.DatasetBase(),
#                        dump_path=manager.DatasetBase()+'data/split_MalFusion_v1.json')
# revertDatasetSplit(dataset='virushare-45',
#                    dump_path=manager.DatasetBase()+'data/split_1.json')
# deleteDatasetSplit(dataset_base=manager.DatasetBase())
###############################################################


# makeDatasetDirStruct(base_path="/home/omnisky/NewAsichurter/ApiData/LargePE-Per35/")


# 分割数据集
################################################################
# man = PathManager(dataset='LargePE-Per35')
# # deleteDatasetSplit(dataset_base=man.DatasetBase())
# splitDatas(src=man.DatasetBase()+'all/',
#            dest=man.DatasetBase()+'train/',
#            ratio=-1,
#            mode='c',
#            is_dir=True)
# splitDatas(src=man.DatasetBase()+'train/',
#            dest=man.DatasetBase()+'validate/',
#            ratio=30,
#            mode='x',
#            is_dir=True)
# splitDatas(src=man.DatasetBase()+'train/',
#            dest=man.DatasetBase()+'test/',
#            ratio=30,
#            mode='x',
#            is_dir=True)
################################################################

# 制作基于下标的数据集
################################################################
for d_type in ['train', 'validate', 'test']:
    manager = PathManager(dataset='virushare-20-3gram-tfidf', d_type=d_type)

    makeDataFile(json_path=manager.Folder(),
                 w2idx_path=manager.WordIndexMap(),
                 seq_length_save_path=manager.FileSeqLen(),
                 data_save_path=manager.FileData(),
                 idx2cls_mapping_save_path=manager.FileIdx2Cls(),
                 num_per_class=20,
                 max_seq_len=300)
################################################################

# renameItemFolder('/home/asichurter/datasets/JSONs/LargePE-100-original/')

# 统计序列长度分布
################################################################
# apiStat('/home/asichurter/datasets/JSONs/HKS/all/',
#         ratio_stairs=[50, 100, 200, 400, 500, 1000, 2000, 5000],
#         dump_report_path=None,#'/home/asichurter/datasets/reports/HKS_3gram_tfidf_api_report.json',#None,#
#         dump_apiset_path=None,#'/home/asichurter/datasets/reports/HKS_3gram_tfidf_api_set.json',#None
#         class_dir=True)
################################################################


# 统计满足数量规模的类别
################################################################
# statSatifiedClasses(pe_path='/home/asichurter/datasets/PEs/virushare_20/all/',
#                     json_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
#                     report_path='/home/asichurter/datasets/reports/virushare-20-original_api_report.json',
#                     stat_stairs=[20,30,40,50],
#                     count_dump_path='/home/asichurter/datasets/reports/virushare-20-original_api_scale.json')
################################################################

# 按照已经知道的满足规模的类进行收集
################################################################
# makeDatasetDirStruct(base_path='/home/asichurter/datasets/JSONs/virushare-20-orginal/')
# collectJsonByClass(pe_path='/home/asichurter/datasets/PEs/virushare_20/all/',
#                    json_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
#                    dst_path='/home/asichurter/datasets/JSONs/virushare-20-orginal/all/',
#                    report_path='/home/asichurter/datasets/reports/virushare-20-original_api_report.json',
#                    num_per_class=20,
#                    selected_classes=["gamevance", "ibryte", "zapchast", "xorer", "installmonetizer", "kovter", "upatre", "pykspa", "lunam", "mepaow", "darkkomet", "browsefox", "urelas", "refroso", "ipamor", "bundlore", "scrinject", "startp", "fakeie", "blacole", "msposer", "soft32downloader", "bettersurf", "dealply", "outbrowse", "psyme", "softpulse", "wajam", "patchload", "dlhelper", "cidox", "4shared", "badur", "fearso", "pirminay", "faceliker", "autoit", "kykymber", "cpllnk", "qqpass", "darbyen", "hijacker", "domaiq", "kido", "fujacks", "redir", "jyfi", "scarsi", "webprefix", "llac", "fosniw", "fbjack", "softcnapp", "getnow", "1clickdownload", "zegost", "gator", "inor", "wonka", "softonic", "nimda", "downloadsponsor", "downloadadmin", "egroupdial", "wabot", "antavmu", "delbar", "zzinfor", "banload", "jeefo", "zbot", "adclicer", "icloader", "reconyc", "vilsel", "installerex", "downloadassistant", "sytro", "sefnit", "staser", "pullupdate", "microfake", "zeroaccess", "somoto", "linkular", "fsysna", "firseria", "loadmoney", "vtflooder", "mydoom", "pophot", "acda", "extenbro", "decdec", "black", "loring", "xtrat", "midia", "nitol", "linkury", "shipup", "gepys", "zvuzona", "urausy", "lineage", "refresh", "yoddos", "iframeref", "mikey", "goredir", "instally", "toggle", "hidelink", "airinstaller", "hicrazyk", "simbot", "conficker", "trymedia", "lipler", "ircbot", "hiloti", "qhost", "buterat", "includer", "iframeinject", "unruy", "directdownloader", "c99shell", "fareit", "windef", "vittalia"])
################################################################

# 将数据集转化为下标形式来减少内存占用
################################################################
# man = PathManager(dataset='HKS-api', d_type='all')
# apiSet = loadJson('D:/datasets/reports/HKS-api_api_set_report.json')
# apis = apiSet['api_set']
# mapping = {name:str(i) for i,name in enumerate(apis)}
# apiSet['api_map'] = mapping
# mappingApiNormalize(json_path=man.Folder(),
#                     mapping=mapping,
#                     is_class_dir=True)
# save back the api mapping
# dumpJson(apiSet, 'D:/datasets/reports/HKS-api_api_set_report.json')
################################################################


# 训练W2V模型(GloVe)
################################################################
# manager = PathManager(dataset='virushare-10-seq', d_type='all')
# seqs = aggregateApiSequences(manager.Folder())
# trainW2Vmodel(seqs,
#               save_matrix_path=manager.WordEmbedMatrix(),
#               save_word2index_path=manager.WordIndexMap(),
#               size=256)
################################################################

# 统计ngram
################################################################
# print('Removing Redundance...')

#
# man = PathManager(dataset='HKS-api', d_type='all')
# removeApiRedundance(json_path=man.Folder(),
#                     class_dir=True)
# print('Stating NGram...')
# ngram_dict = statNGram(parent_path='/home/asichurter/datasets/JSONs/HKS-json/',
#                        window=3,
#                        dict_save_path='/home/asichurter/datasets/reports/HKS_3gram_api_frq.json',
#                        frequency_stairs=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
#                        class_dir=True)
#
# d = loadJson('/home/asichurter/datasets/reports/HKS_3gram_api_frq.json')
#
# print('Converting NGram...')
# convertToNGramSeq(parent_path='/home/asichurter/datasets/JSONs/HKS-json/',
#                   window=3,
#                   ngram_dict=ngram_dict,
#                   ngram_max_num=None,
#                   class_dir=True)
#
# apiStat(path=man.Folder(),
#         dump_report_path='D:/datasets/reports/HKS-api_api_report.json',
#         dump_apiset_path='D:/datasets/reports/HKS-api_api_set_report.json',
#         class_dir=True)
################################################################



# 计算tdidf并且根据该值过滤API
 #################################################################
# num_constraint = 2000
# api_set = loadJson('/home/asichurter/datasets/reports/HKS_3gram_api_set.json')['api_set']
# dict_map = {k:i for i,k in enumerate(api_set)}
# dumpJson(dict_map, '/home/asichurter/datasets/reports/HKS_3gram_tfidf_api_dictmap.json')
# top_k_apis = calTFIDF(dataset_path='/home/asichurter/datasets/JSONs/HKS/all/',
#                       dict_map_path='/home/asichurter/datasets/reports/HKS_3gram_tfidf_api_dictmap.json',
#                       is_class_dir=True,
#                       tfidf_dump_path='/home/asichurter/datasets/reports/HKS_3gram_tfidf_api_val.json',
#                       top_k=num_constraint)
# api_tfidf = loadJson('/home/asichurter/datasets/reports/HKS_3gram_tfidf_api_val.json')
# print('Sorting...')
# api_tfidf = sorted(api_tfidf.items(), key=lambda item:item[1], reverse=True)
# api_list = [api[0] for i,api in enumerate(api_tfidf) if i < num_constraint]
# print('Filtering...')
# filterApiSequence(json_path='/home/asichurter/datasets/JSONs/HKS/all/',
#                   api_list=api_list,
#                   keep_or_filter=False,
#                   is_class_dir=True)
# apiStat('/home/asichurter/datasets/JSONs/HKS/all/',
#         ratio_stairs=[50, 100, 200, 400, 500, 1000, 2000, 5000],
#         dump_report_path=None,#'/home/asichurter/datasets/reports/HKS_3gram_tfidf_api_report.json',#None,#
#         dump_apiset_path='/home/asichurter/datasets/reports/HKS_3gram_tfidf_api_set.json',#None
#         class_dir=True)
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

# json_path = '/home/asichurter/datasets/JSONs/virushare-20-3gram/all/'
# pe_path = '/home/asichurter/datasets/PEs/virushare_20/all/'
#
# for folder in os.listdir(json_path):
#     for item in os.listdir(json_path+folder+'/'):
#         item = item[:-5]
#         if not os.path.exists(pe_path+folder+'/'+item):
#             print(folder+'/'+item, 'not exists in pe path!')