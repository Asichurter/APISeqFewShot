# -*- coding: utf-8

import numpy as np
import torch as t
import torch.nn.functional as F

from scripts.dataset import makeDataFile, makeDatasetDirStruct, splitDatas
from utils.manager import PathManager
from scripts.reshaping import makeMatrixData
from scripts.preprocessing import apiStat, removeApiRedundance, statSatifiedClasses, \
                                    collectJsonByClass, mappingApiNormalize
from models.ProtoNet import IncepProtoNet
from utils.matrix import batchDot
from scripts.embedding import *
from extractors.ngram import statNGram, convertToNGramSeq

from config import generateConfigReport

# 生成报告总结
################################################################
generateConfigReport(dataset='virushare_20', include_result=True)
################################################################


# 制作基于下标的数据集
################################################################
# makeDatasetDirStruct(base_path="/home/asichurter/datasets/JSONs/virushare_20/")
# for d_type in ['train', 'validate', 'test']:
#     manager = PathManager(dataset='virushare_20', d_type=d_type)
#
#     makeDataFile(json_path=manager.Folder(),
#                  w2idx_path=manager.WordIndexMap(),
#                  seq_length_save_path=manager.FileSeqLen(),
#                  data_save_path=manager.FileData(),
#                  num_per_class=20,
#                  max_seq_len=50)
################################################################

# 统计序列长度分布
################################################################
# apiStat('/home/asichurter/datasets/JSONs/jsons-3gram/',
#         ratio_stairs=[500, 1000, 2000, 4000, 5000, 10000, 20000, 50000],
#         dump_report_path=None,#'/home/asichurter/datasets/reports/virushare_3gram_api_report.json',
#         dump_apiset_path=None,#'/home/asichurter/datasets/reports/virushare_3gram_api_set.json',
#         class_dir=False)
################################################################



# 统计满足数量规模的类别
################################################################
# statSatifiedClasses(pe_path='/home/asichurter/datasets/PEs/virushare_20/all/',
#                     json_path='/home/asichurter/datasets/JSONs/jsons-3gram/',
#                     report_path='/home/asichurter/datasets/reports/virushare_3gram_api_report.json',
#                     stat_stairs=[5,10,15,20],
#                     count_dump_path='/home/asichurter/datasets/reports/virushare_3gram_scale_report.json')
################################################################

# 按照已经知道的满足规模的类进行收集
################################################################
# makeDatasetDirStruct(base_path='/home/asichurter/datasets/JSONs/virushare_20/')
# collectJsonByClass(pe_path='/home/asichurter/datasets/PEs/virushare_20/all/',
#                    json_path='/home/asichurter/datasets/JSONs/jsons-3gram/',
#                    dst_path='/home/asichurter/datasets/JSONs/virushare_20/all/',
#                    selected_classes=["ibryte", "zapchast", "xorer", "installmonetizer", "kovter", "lunam", "darkkomet", "urelas", "refroso", "ipamor", "bundlore", "scrinject", "startp", "fakeie", "blacole", "msposer", "soft32downloader", "bettersurf", "dealply", "outbrowse", "psyme", "patchload", "dlhelper", "4shared", "badur", "fearso", "pirminay", "faceliker", "autoit", "kykymber", "cpllnk", "qqpass", "darbyen", "hijacker", "domaiq", "kido", "fujacks", "redir", "jyfi", "scarsi", "webprefix", "llac", "fosniw", "fbjack", "softcnapp", "getnow", "1clickdownload", "gator", "inor", "wonka", "softonic", "nimda", "downloadsponsor", "downloadadmin", "egroupdial", "wabot", "antavmu", "zzinfor", "banload", "jeefo", "zbot", "adclicer", "icloader", "reconyc", "vilsel", "installerex", "downloadassistant", "sytro", "sefnit", "staser", "microfake", "zeroaccess", "somoto", "linkular", "fsysna", "firseria", "loadmoney", "mydoom", "acda", "extenbro", "decdec", "black", "loring", "xtrat", "midia", "shipup", "gepys", "zvuzona", "urausy", "lineage", "refresh", "yoddos", "iframeref", "mikey", "goredir", "instally", "toggle", "hidelink", "airinstaller", "hicrazyk", "simbot", "trymedia", "lipler", "ircbot", "hiloti", "qhost", "buterat", "includer", "iframeinject", "directdownloader", "c99shell", "windef", "vittalia"])
################################################################


# 将数据集转化为下标形式来减少内存占用
################################################################
# apiSet = loadJson('/home/asichurter/datasets/reports/virushare_3gram_api_set.json')['api_set']
# mapping = {name:str(i) for i,name in enumerate(apiSet)}
# mappingApiNormalize(json_path='/home/asichurter/datasets/JSONs/virushare_20/all/',
#                     mapping=mapping,
#                     is_class_dir=True)
################################################################


# 分割数据集
################################################################
# splitDatas(src='/home/asichurter/datasets/JSONs/virushare_20/all/',
#            dest='/home/asichurter/datasets/JSONs/virushare_20/train/',
#            ratio=113,
#            mode='c',
#            is_dir=True)
# splitDatas(src='/home/asichurter/datasets/JSONs/virushare_20/train/',
#            dest='/home/asichurter/datasets/JSONs/virushare_20/validate/',
#            ratio=20,
#            mode='x',
#            is_dir=True)
# splitDatas(src='/home/asichurter/datasets/JSONs/virushare_20/train/',
#            dest='/home/asichurter/datasets/JSONs/virushare_20/test/',
#            ratio=20,
#            mode='x',
#            is_dir=True)
################################################################


# 训练W2V模型(GloVe)
################################################################
# manager = PathManager(dataset='virushare_20', d_type='all')
# seqs = aggregateApiSequences("/home/asichurter/datasets/JSONs/virushare_20/all/")
# trainW2Vmodel(seqs,
#               save_matrix_path=manager.WordEmbedMatrix(),
#               save_word2index_path=manager.WordIndexMap(),
#               size=300)
################################################################

# 统计ngram
################################################################
# removeApiRedundance(json_path='/home/asichurter/datasets/JSONs/jsons-3gram/')
#
# ngram_dict = statNGram(parent_path='/home/asichurter/datasets/JSONs/jsons-3gram/',
#                        window=3,
#                        dict_save_path='/home/asichurter/datasets/JSONs/3gram_dict.json',
#                        frequency_stairs=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
#
# num = int(input('NGram >> '))
#
# convertToNGramSeq(parent_path='/home/asichurter/datasets/JSONs/jsons-3gram/',
#                   window=3,
#                   ngram_dict=ngram_dict,
#                   ngram_max_nu93m=num)
################################################################


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

