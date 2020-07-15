import os
import shutil
import random
import torch as t
import pandas as pd
from time import time
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from utils.error import Reporter
from utils.file import loadJson, dumpJson
from utils.display import printState
from utils.manager import PathManager

magic = 7355608

##########################################################
# 本文件是为了将数据集中的类进行随机抽样移动，可以用于数据集分割
##########################################################
def splitDatas(src, dest, ratio, mode='x', is_dir=False):
    '''
    将生成的样本按比例随机抽样分割，并且移动到指定文件夹下，用于训练集和验证集的制作
    src:源文件夹
    dest:目标文件夹
    ratio:分割比例或者最大数量
    '''
    assert mode in ['c', 'x'], '选择的模式错误，只能复制c或者剪切x'

    All = os.listdir(src)

    if ratio < 0:
        size = len(All)
    elif 1 > ratio > 0:
        size = int(len(All) * ratio)
    else:
        size = ratio
    # size = int(len(All) * ratio) if ratio < 1 else ratio

    assert len(All) >= size, '分割时，总数量没有要求的数量大！'

    random.seed(time() % magic)
    samples_names = random.sample(All, size)
    num = 0
    for item in tqdm(All):
        if item in samples_names:
            num += 1
            path = src + item
            if mode == 'x':
                shutil.move(path, dest)
            else:
                if is_dir:
                    shutil.copytree(src=path, dst=dest+item)
                else:
                    shutil.copy(src=path, dst=dest)


##########################################################
# 本函数主要用于数据集的文件生成。
# 用于根据已经按类分好的JSON形式数据集，根据已经生成的嵌入矩阵和
# 词语转下标表来将数据集整合，token替换为对应的词语下标序列，同时pad，最后
# 将序列长度文件和数据文件进行存储的总调用函数。运行时要检查每个类的样本
# 数，也会按照最大序列长度进行截断。
##########################################################
def makeDataFile(json_path,
                 w2idx_path,
                 seq_length_save_path,
                 data_save_path,
                 num_per_class,
                 idx2cls_mapping_save_path=None,
                 max_seq_len=600):

    data_list = []
    folder_name_mapping = {}

    printState('Loading config data...')
    word2index = loadJson(w2idx_path)

    printState('Read main data...')
    for cls_idx, cls_dir in tqdm(enumerate(os.listdir(json_path))):
        class_path = json_path + cls_dir + '/'

        assert num_per_class == len(os.listdir(class_path)), \
            '数据集中类%s的样本数量%d与期望的样本数量不一致！'%\
            (cls_dir, len(os.listdir(class_path)), num_per_class)

        for item in os.listdir(class_path):
            report = loadJson(class_path + item)
            apis = report['apis']
            data_list.append(apis)          # 添加API序列

        folder_name_mapping[cls_idx] = cls_dir

        # label_list += [cls_idx] * num_per_class     # 添加一个类的样本标签

    printState('Converting...')
    data_list = convertApiSeq2DataSeq(data_list,
                                      word2index,
                                      max_seq_len)      # 转化为嵌入后的数值序列列表

    seq_length_list = {i:len(seq) for i,seq in enumerate(data_list)}   # 数据的序列长度

    data_list = pad_sequence(data_list, batch_first=True, padding_value=0)  # 数据填充0组建batch

    # 由于pad函数是根据输入序列的最大长度进行pad,如果所有序列小于最大长度,则有可能出现长度
    #　不一致的错误
    if data_list.size(1) < max_seq_len:
        padding_size = max_seq_len - data_list.size(1)
        zero_paddings = t.zeros((data_list.size(0),padding_size))
        data_list = t.cat((data_list,zero_paddings),dim=1)

    printState('Dumping...')
    dumpJson(seq_length_list, seq_length_save_path)     # 存储序列长度到JSON文件
    if idx2cls_mapping_save_path is not None:
        dumpJson(folder_name_mapping, idx2cls_mapping_save_path)
    t.save(data_list, data_save_path)                   # 存储填充后的数据文件

    printState('Done')

##########################################################
# 本函数是makeDataFile函数的调用函数，主要用于截断序列，同时根据下标
# 表和嵌入矩阵将token替换为词语下标（替换为向量的过由Embedding完成）
##########################################################
def convertApiSeq2DataSeq(api_seqs, word2idx, max_size):
    data_seq = []

    for seq in api_seqs:
        appended_seq = []

        if len(seq) > max_size:         # 根据规定的最大序列长度截断序列
            seq = seq[:max_size]

        for i,api in enumerate(seq):
            appended_seq.append(word2idx[api])     # 将API token转化为下标
        data_seq.append(t.Tensor(appended_seq))

    return data_seq

##########################################################
# 本函数用于创建数据集的文件夹结构
##########################################################
def makeDatasetDirStruct(base_path):
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    os.mkdir(base_path + 'all/')
    os.mkdir(base_path + 'train/')
    os.mkdir(base_path + 'validate/')
    os.mkdir(base_path + 'test/')
    os.mkdir(base_path + 'models/')

    os.mkdir(base_path + 'data/')
    os.mkdir(base_path + 'data/train/')
    os.mkdir(base_path + 'data/validate/')
    os.mkdir(base_path + 'data/test/')
    os.mkdir(base_path + 'doc/')

    printState('Done')

##########################################################
# 本函数用于将训练集，验证集和测试集的数据集分割详情保存到JSON文件
# 以便复现出实验结果
##########################################################
def dumpDatasetSplitStruct(base_path, dump_path):
    dump = {}
    for split in ['train', 'validate', 'test']:
        print(split)
        folders = []

        for folder in os.listdir(base_path+split+'/'):
            folders.append(folder)

        dump[split] = folders

    dumpJson(dump, dump_path)
    print('-- Done --')


##########################################################
# 本函数用于删除原有数据集train，validate和test文件夹中的数据
##########################################################
def deleteDatasetSplit(dataset_base):
    for typ in ['train','validate','test']:
        print(typ)
        os.system('rm -rf {path}/*'.format(path=dataset_base + typ))

##########################################################
# 本函数用于将训练集，验证集和测试集的数据集分割详情根据保存的JSON文件
# 还原到数据集分隔文件夹中。
##########################################################
def revertDatasetSplit(dataset, dump_path):
    man = PathManager(dataset)
    split_dump = loadJson(dump_path)

    deleteDatasetSplit(man.DatasetBase())

    for typ in ['train', 'validate', 'test']:

        # delete the existed split
        # os.system('rm -rf {path}/*'.format(path=man.DatasetBase()+typ))

        print(typ)
        for folder in split_dump[typ]:
            shutil.copytree(src=man.DatasetBase()+'all/'+folder+'/',
                            dst=man.DatasetBase()+typ+'/'+folder+'/')

    print('-- Done --')

######################################################
# 将Cuckoo分析后并且cleanup过的文件夹和内部文件的名称命名为
#　file的名称
######################################################
def renameCuckooFolders(json_path):
    reporter = Reporter()

    for folder in tqdm(os.listdir(json_path)):
        try:
            report = loadJson(json_path+folder+'/report.json')
            name = report['target']['file']['name']

            os.rename(json_path+folder+'/report.json', json_path+folder+'/%s.json'%name)
            os.rename(json_path+folder, json_path+name)

            reporter.logSuccess()

        except Exception as e:
            reporter.logError(entity=folder, msg=str(e))
            continue

    reporter.report()

#####################################################
# 制作包含原PE文件的占位文件结构,以便在按类分类样本时使用
#####################################################
def makePEDummy(pe_path, dst_path):

    for folder in tqdm(os.listdir(pe_path)):
        if os.path.exists(dst_path+folder+'/'):
            raise RuntimeError('%s已经存在!'%folder)
        else:
            os.mkdir(dst_path+folder+'/')

        for item in os.listdir(pe_path+folder+'/'):
            with open(dst_path+folder+'/'+item, 'w') as f:
                pass


######################################################
# 将vt的JSON报告移动到其PE文件对应的类文件夹中
######################################################
def moveJsonToGroup(src_path,
                    pe_map_path,
                    dst_path):

    item_group_mapper = {
        item: group for group in os.listdir(pe_map_path) for item in os.listdir(pe_map_path+group+'/')
    }

    for folder in tqdm(os.listdir(src_path)):
        group = item_group_mapper[folder]
        shutil.copy(src_path+folder+'/'+folder+'.json',
                    dst_path+group+'/'+folder+'.json')


#####################################################
# 根据已有的PE文件,移除不存在对应PE文件的JSON扫描文件
#####################################################
def removeNotExistItem(index_path, item_path):
    indexed_items = os.listdir(index_path)

    for item in tqdm(os.listdir(item_path)):
        item_name = item.split('.')[0]
        if item_name not in indexed_items:
            os.remove(item_path+item)

    item_amount = len(os.listdir(item_path))
    print('After removing, %d items remain'%item_amount)


#####################################################
# 将个体文件夹和其中的报告命名为样本的名称
#####################################################
def renameItemFolder(json_path):

    for folder in tqdm(os.listdir(json_path)):

        report = loadJson(json_path + folder + '/report.json')
        name = report['target']['file']['name']

        os.rename(json_path + folder + '/report.json', json_path + folder + '/%s.json'%name)
        os.rename(json_path+folder+'/', json_path+name+'/')


if __name__ == '__main__':
    pass
    # splitDatas(src='D:/peimages/JSONs/virushare_20/train/',
    #            dest='D:/peimages/JSONs/virushare_20/test/',
    #            ratio=30,
    #            mode='x',
    #            is_dir=True)

    # a = [t.Tensor([[1, 2], [3, 4], [5, 6]]), t.Tensor([[11, 12], [13, 14]]), t.Tensor([[21, 22]])]
    # data_write_csv('data.csv', a)

