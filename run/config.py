import sys
import os
import shutil

from utils.file import deleteDir

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
