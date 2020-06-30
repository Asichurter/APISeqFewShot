import os
import zipfile
from tqdm import tqdm

from utils.error import Reporter


def extractZipFile(src, dst, psw):
    psw = psw.encode()
    zip = zipfile.ZipFile(src)
    zip.extractall(path=dst, pwd=psw)   # 解压到目标文件夹
    zip.close()

################################################
# 根据统一的密码,将一个目录下的所有文件解压到另一个目录中
################################################
def extractAllZipFile(src_dir, dst_dir, psw):
    reporter = Reporter()

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for item in tqdm(os.listdir(src_dir)):
        try:
            extractZipFile(src=src_dir+item,
                           dst=dst_dir,
                           psw=psw)
            reporter.logSuccess()

        except RuntimeError as e:
            reporter.logError(entity=item, msg=str(e))

    reporter.report()

if __name__ == '__main__':
    extractAllZipFile(src_dir='/home/asichurter/datasets/PEs/wudi/ziped/',
                      dst_dir='/home/asichurter/datasets/PEs/wudi/unziped/',
                      psw='infected')
