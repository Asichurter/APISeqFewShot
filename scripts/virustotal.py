import os
import shutil
from tqdm import tqdm
import time
import requests
import sys

from utils.file import loadJson, dumpJson

scan_url = 'https://www.virustotal.com/vtapi/v2/file/scan'
report_url = 'https://www.virustotal.com/vtapi/v2/file/report'
apikey = 'c424abc9c8d7102cfaf9cf2d8f01fb95f4ddfd81a563d6e07738fa960b501d87'


###############################################
# 将所有样本提取到一个文件夹中
###############################################
def collectPEasExistingDataset(json_path,
                               pe_path,
                               dst_path,
                             is_class_dir=True):
    for folder in tqdm(os.listdir(json_path)):
        if is_class_dir:
            items = os.listdir(json_path+folder+'/')
        else:
            items = [folder+'.json']

        for item in items:
            pe_item_name = '.'.join(item.split('.')[:-1])
            shutil.copy(pe_path+folder+'/'+pe_item_name,
                        dst_path+pe_item_name)

def vtScan(folder_path, json_save_path, scan_num=20000,
           timeout=600):
    scan_params = {'apikey': apikey}

    start_index = len(os.listdir(json_save_path))
    end_index = min(start_index+scan_num,len(os.listdir(folder_path)))

    print('Begin to scan...')
    samples_list = os.listdir(folder_path)
    last_stamp = time.time()
    while start_index < end_index:
        print(start_index+1,'/',end_index)
        f = samples_list[start_index]
        if (os.path.exists(json_save_path + f + '.json') and os.path.getsize(json_save_path + f + '.json') != 0):
            start_index += 1
            last_stamp = time.time()
            continue

        files_cfg = {'file': ('test', open(folder_path + f, 'rb'))}

        try:
            print('scanning...')
            response = requests.post(scan_url, files=files_cfg, params=scan_params,
                                     timeout=timeout)
        except Exception as e:
            print(f, ': api request exceeds!', ' error:', str(e))
            print('waiting...')
            time.sleep(10)
            continue

        scan_info = response.json()
        report_params = {'apikey': apikey, 'resource': scan_info['md5']}
        try:
            print('fetching report...')
            report = requests.get(report_url, params=report_params, timeout=timeout)
            report = report.json()  # ['scans']
        except BaseException as e:
            print(f, ': api request exceeds!', ' error:', str(e))
            print('waiting...')
            time.sleep(10)
            continue

        # print(report)
        print(report['verbose_msg'])
        if report['response_code'] == 1:
            dumpJson(report, '%s.json' % (json_save_path + f))
        else:
            sys.stderr.write('%s wrong response code %d' % (f, report['response_code']))

        print('time consuming: %.2f' % (time.time() - last_stamp))
        last_stamp = time.time()

        start_index += 1

        time.sleep(1)

if __name__ == '__main__':
    collectPEasExistingDataset(json_path='/home/asichurter/datasets/JSONs/LargePE-80/all/',
                               pe_path='/home/asichurter/datasets/PEs/LargePE-100/data_folders/',
                               dst_path='/home/asichurter/datasets/JSONs/LargePE-80-all-temp/',
                               is_class_dir=True)
    # vtScan(folder_path='/home/asichurter/datasets/PEs/temp/',
    #        json_save_path='/home/asichurter/datasets/JSONs/temp/')

