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
                                    collectJsonByClass, mappingApiNormalize, filterApiSequence
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
# man = PathManager(dataset='virushare-20-3gram')
# deleteDatasetSplit(dataset_base=man.DatasetBase())
# splitDatas(src=man.DatasetBase()+'all/',
#            dest=man.DatasetBase()+'train/',
#            ratio=-1,
#            mode='c',
#            is_dir=True)
# splitDatas(src=man.DatasetBase()+'train/',
#            dest=man.DatasetBase()+'validate/',
#            ratio=20,
#            mode='x',
#            is_dir=True)
# splitDatas(src=man.DatasetBase()+'train/',
#            dest=man.DatasetBase()+'test/',
#            ratio=20,
#            mode='x',
#            is_dir=True)
################################################################

# 制作基于下标的数据集
################################################################
# makeDatasetDirStruct(base_path="/home/asichurter/datasets/JSONs/virushare-20-h3gram/")
# for d_type in ['train', 'validate', 'test']:
#     manager = PathManager(dataset='virushare-20-h3gram', d_type=d_type)
#
#     makeDataFile(json_path=manager.Folder(),
#                  w2idx_path=manager.WordIndexMap(),
#                  seq_length_save_path=manager.FileSeqLen(),
#                  data_save_path=manager.FileData(),
#                  num_per_class=20,
#                  max_seq_len=200)
################################################################

# 统计序列长度分布
################################################################
# apiStat('/home/asichurter/datasets/JSONs/jsons-3gram/',
#         ratio_stairs=[500, 1000, 2000, 4000, 5000, 10000, 20000, 50000],
#         dump_report_path='/home/asichurter/datasets/reports/virushare-20_3gram_api_report.json',#None,#
#         dump_apiset_path='/home/asichurter/datasets/reports/virushare-20_3gram_api_set.json',#None
#         class_dir=False)
################################################################



# 统计满足数量规模的类别
################################################################
# statSatifiedClasses(pe_path='/home/asichurter/datasets/PEs/virushare_20/all/',
#                     json_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
#                     report_path='/home/asichurter/datasets/reports/virushare-10_3gram_api_report.json',
#                     stat_stairs=[5,10,15,20],
#                     count_dump_path='/home/asichurter/datasets/reports/virushare-10_3gram_scale_report.json')
################################################################

# 按照已经知道的满足规模的类进行收集
################################################################
# makeDatasetDirStruct(base_path='/home/asichurter/datasets/JSONs/virushare-10-3gram/')
# collectJsonByClass(pe_path='/home/asichurter/datasets/PEs/virushare_20/all/',
#                    json_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
#                    dst_path='/home/asichurter/datasets/JSONs/virushare-10-3gram/all/',
#                    report_path='/home/asichurter/datasets/reports/virushare-10_3gram_api_report.json',
#                    num_per_class=10,
#                    selected_classes=["winactivator", "gamevance", "ibryte", "zapchast", "xorer", "yakes", "palevo", "banbra", "installmonetizer", "menti", "convertad", "kovter", "dorifel", "upatre", "fakerean", "stegvob", "swizzor", "zlob", "sinowal", "pykspa", "lunam", "bancos", "mepaow", "darkkomet", "filetour", "sality", "simda", "browsefox", "brontok", "klone", "urelas", "dapato", "staget", "refroso", "vobfus", "ipamor", "delfinject", "bundlore", "scrinject", "imali", "startp", "ngrbot", "fakeie", "berbew", "blacole", "msposer", "soft32downloader", "bettersurf", "archsms", "dealply", "outbrowse", "psyme", "koutodoor", "dybalom", "softpulse", "wajam", "bladabindi", "chekafev", "patchload", "dlhelper", "cidox", "ramnit", "4shared", "badur", "fearso", "disfa", "pirminay", "faceliker", "mabezat", "geral", "autoit", "cycbot", "kykymber", "slugin", "cpllnk", "qqpass", "hotbar", "wapomi", "darbyen", "poison", "lmir", "viking", "multiplug", "gamehack", "hijacker", "domaiq", "kido", "boaxxe", "fujacks", "redir", "framer", "kolabc", "parite", "jyfi", "pasta", "scarsi", "fraudload", "virlock", "pcclient", "webprefix", "llac", "agentb", "fosniw", "cosmu", "fbjack", "nsanti", "softcnapp", "getnow", "1clickdownload", "zegost", "monder", "mudrop", "gator", "atraps", "inor", "socks", "wonka", "vundo", "xpaj", "softonic", "swisyn", "pakes", "nimda", "bamital", "opencandy", "farfli", "nilage", "downloadsponsor", "rbot", "downloadadmin", "egroupdial", "pioneer", "wabot", "rodecap", "antavmu", "beebone", "techsnab", "delbar", "crytex", "zzinfor", "banload", "jeefo", "zbot", "adclicer", "tdss", "ldpinch", "icloader", "spyeye", "reconyc", "vilsel", "installerex", "turkojan", "downloadassistant", "sytro", "onlinegames", "delf", "sefnit", "rebhip", "hupigon", "winwebsec", "staser", "pullupdate", "relnek", "usteal", "microfake", "zeroaccess", "somoto", "linkular", "fsysna", "luder", "hlux", "scar", "karagany", "gamarue", "lollipop", "firseria", "daws", "loadmoney", "vtflooder", "mydoom", "acda", "extenbro", "decdec", "buzus", "black", "fraudrop", "loring", "xtrat", "chifrax", "ganelp", "midia", "amonetize", "installcore", "zusy", "nitol", "shipup", "gepys", "installbrain", "medfos", "zvuzona", "sohanad", "blacoleref", "urausy", "flystudio", "lineage", "crossrider", "refresh", "yoddos", "iframeref", "resur", "mikey", "shiz", "kelihos", "goredir", "instally", "toggle", "hidelink", "airinstaller", "megasearch", "malex", "hicrazyk", "simbot", "magania", "picsys", "conficker", "trymedia", "razy", "lipler", "ircbot", "hiloti", "vmprotbad", "qhost", "eorezo", "buterat", "barys", "dorkbot", "smartfortress", "renos", "fakefolder", "includer", "iframeinject", "bifrose", "bublik", "unruy", "directdownloader", "mira", "kuluoz", "c99shell", "fareit", "mediaget", "windef", "vittalia", "neshta"])
################################################################


# 将数据集转化为下标形式来减少内存占用
################################################################
# apiSet = loadJson('/home/asichurter/datasets/reports/virushare-20_3gram_api_set.json')
# apis = apiSet['api_set']
# mapping = {name:str(i) for i,name in enumerate(apis)}
# apiSet['api_map'] = mapping
# mappingApiNormalize(json_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
#                     mapping=mapping,
#                     is_class_dir=False)
# # save back the api mapping
# dumpJson(apiSet, '/home/asichurter/datasets/reports/virushare-20_3gram_api_set.json')
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
# removeApiRedundance(json_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
#                     class_dir=False)
#
# man = PathManager(dataset='virushare-20-h3gram', d_type='all')
# ngram_dict = statNGram(parent_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
#                        window=3,
#                        dict_save_path='/home/asichurter/datasets/reports/virushare-20_3gram_tfidf_api_set.json',
#                        frequency_stairs=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
#                        class_dir=False)
#
# num = int(input('NGram >> '))
# #
# d = loadJson('/home/asichurter/datasets/reports/virushare-20_h3gram_api_freq.json')
#
# convertToNGramSeq(parent_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
#                   window=3,
#                   ngram_dict=ngram_dict,
#                   ngram_max_num=None,
#                   class_dir=False)
################################################################

# api_set = loadJson('/home/asichurter/datasets/reports/virushare-20_3gram_tfidf_api_set.json')
# dict_map = {k:i for i,k in enumerate(api_set)}
# dumpJson(dict_map, '/home/asichurter/datasets/reports/virushare-20_3gram_tfidf_api_dictmap.json')
# top_k_apis = calTFIDF(dataset_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
#                       dict_map_path='/home/asichurter/datasets/reports/virushare-20_3gram_tfidf_api_dictmap.json',
#                       is_class_dir=False,
#                       tfidf_dump_path='/home/asichurter/datasets/reports/virushare-20_3gram_tfidf_api_val.json',
#                       top_k=2000)

api_tfidf = loadJson('/home/asichurter/datasets/reports/virushare-20_3gram_tfidf_api_val.json')
api_tfidf = sorted(api_tfidf.items(), key=lambda item:item[1], reverse=True)
api_list = [api[0] for i,api in enumerate(api_tfidf) if i < 2000]
filterApiSequence(json_path='/home/asichurter/datasets/JSONs/jsons - 副本(复件)/',
                  api_list=api_list,
                  keep_or_filter=False)

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
