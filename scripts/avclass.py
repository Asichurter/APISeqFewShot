import os
import json
import shutil
from tqdm import tqdm
import numpy as np
import random as rd

from utils.error import Reporter
from utils.file import loadJson, dumpJson
from utils.magic import magicSeed

######################################################
# 将已有的json数据集中的数据全部收集复制到一个文件夹中
######################################################
def collectJsonFromExistingDataset(json_path, dst_path,
                                   is_class_dir=True):

    reporter = Reporter()

    for folder in tqdm(os.listdir(json_path)):
        if is_class_dir:
            items = os.listdir(json_path+folder+'/')
        else:
            items = [folder+'.json']

        for item in items:
            if not os.path.exists(dst_path+item):
                shutil.copy(json_path+folder+'/'+item, dst_path+item)
                reporter.logSuccess()
            else:
                reporter.logError(entity=folder+'/'+item, msg="Duplicate exists")

    reporter.report()

######################################################
# 假设JSON报告与数据文件同名(数据扩展名由ext_name指定),利用
# virustotal报告中的MD5值来重新命名报告与数据文件
######################################################
def renameItemsByMD5(json_path, item_path, ext_name=''):
    reporter = Reporter()

    md5s = []

    for json_item in tqdm(os.listdir(json_path)):
        report = loadJson(json_path+json_item)
        md5 = report['md5']

        if md5 in md5s:
            reporter.logWarning(entity=json_item, msg='MD5重复')
            success_flag = False

        else:
            md5s.append(md5)
            success_flag = True

        filename = '.'.join(json_item.split('.')[:-1])

        os.rename(json_path+json_item, json_path+md5+'.json')   # 重命名json报告
        os.rename(item_path+filename+ext_name, item_path+md5+ext_name)   # 重命名数据文件

        if success_flag:
            reporter.logSuccess()

    reporter.report()


######################################################
# 由于获取报告时JSON文件的indent不为None,本方法将virustotal报
# 告的JSON文件的indent重新设置为None以便于avclass输入
######################################################
def redumpJsonReport(report_path):
    for item in tqdm(os.listdir(report_path)):
        report = loadJson(report_path+item)
        dumpJson(report, report_path+item, indent=None)


######################################################
# 运行avclass获取标签数据文件
######################################################
def runAvclass(report_path,
               label_out_path,
               avclass_path):
    command = f'{avclass_path} -vtdir {report_path} > {label_out_path}'
    os.system(command)


######################################################
# 根据生成的avclass标签文件,将数据文件移动到对应的类文件夹中
######################################################
def moveToFolderByClass(label_file_path, src_path, dst_path, ext_name=''):
    families = set()

    os.mkdir(dst_path+'SINGLETON/')

    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        for i,line in tqdm(enumerate(lines)):
            line = line.replace('\t', ' ')
            line = line.replace('\n', '')
            items = line.split(' ')
            md5, label = items
            # SINGLETON单独一个文件夹
            if label.startswith('SINGLETON'):
                shutil.move(src_path+md5+ext_name, dst_path+'SINGLETON/'+md5+ext_name)
                continue
            # 若label第一次出现，则在目标处新建一个类文件夹
            if label not in families:
                families.add(label)
                if not os.path.exists(dst_path+label+'/'):
                    os.mkdir(dst_path+label+'/')
            try:
                shutil.move(src_path + md5 + ext_name, dst_path + label + '/' + md5 + ext_name)
            except FileNotFoundError:
                continue

######################################################
# 统计收集到的类文件夹中数量的规模
######################################################
def statDatasetScale(data_path, stairs=[]):
    nums = []
    for folder in os.listdir(data_path):
        nums.append(len(os.listdir(data_path+folder+'/')))
    nums = np.array(nums)

    for stair in stairs:
        count = (nums >= stair).sum()
        print('*'*50)
        print('超过 %d 个样本的类别数: %d'%(stair, count))
        print('*'*50)


######################################################
# 指定数量规模,将满足数量规模的类,抽样指定数量的数据收集到新的文
# 件夹中
######################################################
def collectScaleClasses(data_path, dst_path, num_per_class=50,
                        exception=['SINGLETON']):
    for folder in tqdm(os.listdir(data_path)):
        if folder in exception:
            continue

        item_list = os.listdir(data_path+folder+'/')
        if len(item_list) >= num_per_class:
            rd.seed(magicSeed())
            candidate_list = rd.sample(item_list, num_per_class)
            os.mkdir(dst_path+folder+'/')

            for item in candidate_list:
                shutil.copy(data_path+folder+'/'+item,
                            dst_path+folder+'/'+item)

if __name__ == '__main__':
    a = 0
    # collectJsonFromExistingDataset(json_path='/home/asichurter/datasets/JSONs/LargePE-80/all/',
    #                                dst_path='/home/asichurter/datasets/JSONs/LargePE-80-vt/all-json/',
    #                                is_class_dir=True)
    # renameItemsByMD5(json_path='/home/asichurter/datasets/JSONs/LargePE-80-vtreport/',
    #                  item_path='/home/asichurter/datasets/JSONs/LargePE-80-vt/all-json/',
    #                  ext_name='.json')
    # redumpJsonReport(report_path='/home/asichurter/datasets/JSONs/LargePE-80-vtreport/')
    # runAvclass(report_path='/home/asichurter/datasets/JSONs/LargePE-80-vtreport/',
    #            label_out_path='/home/asichurter/datasets/JSONs/LargePE-80-vt/all-labels.txt',
    #            avclass_path='/home/asichurter/codes/avclass-master/avclass_labeler.py')
    # moveToFolderByClass(label_file_path='/home/asichurter/datasets/JSONs/LargePE-80-vt/all-labels.txt',
    #                     src_path='/home/asichurter/datasets/JSONs/LargePE-80-vt/all-json/',
    #                     dst_path='/home/asichurter/datasets/JSONs/LargePE-80-vt/category/',
    #                     ext_name='.json')
    # statDatasetScale(data_path='/home/asichurter/datasets/JSONs/LargePE-80-vt/category/',
    #                  stairs=[10*i for i in range(2,9)])
    # collectScaleClasses(data_path='/home/asichurter/datasets/JSONs/LargePE-80-vt/category/',
    #                     dst_path='/home/asichurter/datasets/JSONs/LargePE-80-vt/all/',
    #                     num_per_class=50)
    # c = 1
    # for f in os.listdir('/home/asichurter/datasets/JSONs/LargePE-80-vt/category/'):
    #     if len(os.listdir('/home/asichurter/datasets/JSONs/LargePE-80-vt/category/'+f)) >= 50:
    #         print(c)
    #         c += 1

