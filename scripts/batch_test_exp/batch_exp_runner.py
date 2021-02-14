import os
import sys
import pynvml
import threading
import time

pynvml.nvmlInit()

sys.path.append('../')

from utils.file import loadJson, dumpJson

dataset_list = ['HKS']
seq_len_list = list(range(50, 1050, 50))
n_list = [10]
max_iter = 3000
cuda_device = 1


def set_config_cuda_device(doc, cuda_idx):
    doc['deviceId'] = cuda_idx
    return doc

def set_config_seq_len(doc, len_):
    doc['modelParams']['max_seq_len'] = len_
    return doc


def set_config_n(doc, n):
    doc['n'] = n
    return doc


def set_config_max_iter(doc, max_iter):
    doc['trainingEpoch'] = max_iter
    return doc


def set_config_dataset(doc, dataset):
    doc['dataset'] = dataset
    return doc


def get_gpu_mem_used_MB(cuda_idx):
    handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_idx)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / 1024 / 1024


def gen_data_files(dataset, max_seq_len):
    code = os.system(f'python ../scripts/batch_test_exp/gen_data_files.py -d {dataset} -l {max_seq_len}')
    assert code == 0, 'Fail to run gen_data_files'


def train():
    code = os.system('python ../run/run.py -v -1')
    assert code == 0, 'Fail to run train'


def test():
    code = os.system('python ../run/test.py -v 1')      # todo: 不verbose时，n=10的slice time有异常偏低
    assert code == 0, 'Fail to run test'


def start_train():
    train_thread = threading.Thread(None, target=train, name='train_thread')
    train_thread.start()
    return train_thread


def start_test():
    test_thread = threading.Thread(None, target=test, name='test_thread')
    test_thread.start()
    return test_thread


for dataset in dataset_list:
    for n in n_list:
        for seq_len in seq_len_list:

            print('\n\n')
            print(f'{dataset} n={n} seq_len={seq_len}')
            print('*' * 50)

            print('Prepare config...')
            run_config = loadJson('../run/runConfig.json')
            run_config = set_config_max_iter(run_config, max_iter)
            run_config = set_config_n(run_config, n)
            run_config = set_config_seq_len(run_config, seq_len)
            run_config = set_config_dataset(run_config, dataset)
            run_config = set_config_cuda_device(run_config, cuda_device)
            dumpJson(run_config, '../run/runConfig.json')

            stat = loadJson('../scripts/batch_test_exp/run_stat.json')
            info = {
                'dataset': dataset,
                'n': n,
                'seq_len': seq_len,
                'gpu_memory': None,
                'slice_time': None
            }

            print('Generate data files...')
            gen_data_files(dataset, seq_len)

            print('Run training...')
            train_thread = start_train()
            train_thread.join()

            print('Run testing...')
            test_config = loadJson('../run/testConfig.json')
            test_config = set_config_dataset(test_config, dataset)
            test_config = set_config_n(test_config, n)
            test_config = set_config_cuda_device(test_config, cuda_device)
            dumpJson(test_config, '../run/testConfig.json')

            test_thread = start_test()
            # 等待测试进行20s稳定后进行显存消耗检测
            time.sleep(8)
            info['gpu_memory'] = get_gpu_mem_used_MB(cuda_device)
            stat.append(info)
            dumpJson(stat, '../scripts/batch_test_exp/run_stat.json')
            test_thread.join()

            print('done')





