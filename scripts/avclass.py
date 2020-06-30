import os
import json
import shutil
from tqdm import tqdm

from utils.file import loadJson

def renameItemsByMD5(json_path, item_path):
    for json_item in tqdm(os.listdir(json_path)):
        report = loadJson(json_path+json_item)
        md5 = report['md5']

        filename = json_item.split('.')[0]

        os.rename(json_path+json_item, json_path+md5+'.json')   # 重命名json报告
        os.rename(item_path+filename, item_path+md5)        # 重命名二进制文件

def moveToFolderByClass(label_file_path, src_path, dst_path):
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
                shutil.move(src_path+md5, dst_path+'SINGLETON/'+md5)
                continue
            # 若label第一次出现，则在目标处新建一个类文件夹
            if label not in families:
                families.add(label)
                if not os.path.exists(dst_path+label+'/'):
                    os.mkdir(dst_path+label+'/')
            try:
                shutil.move(src_path + md5, dst_path + label + '/' + md5)
            except FileNotFoundError:
                continue

if __name__ == '__main__':
    # renameItemsByMD5(json_path='/home/asichurter/datasets/PEs/wudi/result/',
    #                  item_path='/home/asichurter/datasets/PEs/wudi/unziped/')
    # moveToFolderByClass(label_file_path='/home/asichurter/datasets/PEs/wudi/labels.txt',
    #                     src_path='/home/asichurter/datasets/PEs/wudi/unziped/',
    #                     dst_path='/home/asichurter/datasets/PEs/wudi/category/')
    c = 1
    for f in os.listdir('/home/asichurter/datasets/PEs/wudi/category/'):
        if len(os.listdir('/home/asichurter/datasets/PEs/wudi/category/'+f)) >= 20:
            print(c)
            c += 1

