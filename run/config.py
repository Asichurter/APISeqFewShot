import sys
import os
import shutil
import logging

from tqdm import tqdm

from utils.file import deleteDir, loadJson, dumpJson
from utils.manager import PathManager

###########################################
# 本函数用于将当前项目路径添加到扫描选项中以便可以
# 导入其他模块。可以使用depth参数来调节运行路径相对
# 项目路径的深度，默认为1
###########################################
def appendProjectPath(depth=1):
    pwd = os.getcwd()  # 要正常运行，运行路径必须与本文件相同
    pwd = repr(pwd).replace('\\\\', '/')[1:-1*depth]  # 替换双斜杠
    projectPath = pwd.split('/')
    projectPath = '/'.join(projectPath[:-1])  # 获取项目路径父路径
    sys.path.append(projectPath)  # 添加当前项目路径到包搜索路径中


def saveConfigFile(folder_path):
    # 若目前的version已经存在，则删除之
    if os.path.exists(folder_path):
        deleteDir(folder_path)
    os.mkdir(folder_path)
    # 复制运行配置文件
    shutil.copy('./runConfig.json', folder_path + 'config.json')


###########################################################
# 从数据集对应的doc文件夹中读取每一次运行的config和测试结果，整理到一
# 个单独的JSON文件中以便浏览
###########################################################
def generateConfigReport(dataset, include_result=False):
    mng = PathManager(dataset)

    report = {}

    for doc_dir in os.listdir(mng.DocBase()):
        config_path = mng.DocBase() + doc_dir + '/'

        try:
            cfg = loadJson(config_path + 'config.json')

            report[int(cfg['version'])] = {
                '__model': cfg['modelName'],                # 添加下划线便于排序
                '_k-q-qk': '-'.join([str(cfg['k']),
                                    str(cfg['n']),
                                    str(cfg['qk'])]),
                'desc': cfg['description']
            }

            if include_result:
                res = loadJson(config_path + 'testResult.json')
                report[int(cfg['version'])]['results'] = res['results']

        except Exception as e:
            logging.warning('Error occurred when process %s: %s'%
                            (doc_dir, str(e)))

    dumpJson(report, mng.DatasetBase()+'summary.json', sort=True)