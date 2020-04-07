import os
from tqdm import tqdm

from utils.file import loadJson, dumpJson
from utils.apis import strlistToStr
from utils.display import printBulletin, printState
from scripts.embedding import aggregateApiSequences
from utils.error import Reporter

def statNGram(parent_path, window=3,
              dict_save_path=None,          # NGram频率的保存
              frequency_stairs=[]):         # 频率阶梯，必须从小到大排列，统计超过该频率需要的最少NGram个数

    reporter = Reporter()

    ngram_dict = {}
    total_cnt = 0

    printState('Counting...')
    for folder in tqdm(os.listdir(parent_path)):
        folder_path = parent_path + folder + '/'

        try:
            seq = loadJson(folder_path + folder + '.json')['apis']

            for i in range(len(seq)-window):
                ngram = strlistToStr(seq[i:i+window])

                total_cnt += 1
                if ngram not in ngram_dict:
                    ngram_dict[ngram] = 1
                else:
                    ngram_dict[ngram] += 1

            reporter.logSuccess()

        except Exception as e:
            reporter.logError(entity=folder, msg=str(e))
            continue

    printState('Processing...')

    # 按照频率降序排列
    ngram_dict = dict(sorted(ngram_dict.items(), key=lambda x: x[1], reverse=True))

    # 频率归一化
    for k in ngram_dict.keys():
        ngram_dict[k] = ngram_dict[k] / total_cnt

    if dict_save_path is not None:
        dumpJson(ngram_dict, dict_save_path)

    # 统计频率分布
    f_accum = 0.
    idx = 0
    keys = list(ngram_dict.keys())
    max_len = len(keys)
    for f_stair in frequency_stairs:
        while f_accum < f_stair and idx < max_len:
            f_accum += ngram_dict[keys[idx]]
            idx += 1
        printBulletin('%f:   %d NGrams'%(f_stair, idx+1))

    printBulletin('Total: %d NGrams'%len(ngram_dict))

    return ngram_dict



def convertToNGramSeq(parent_path, window=3,
                      ngram_dict=None,              # 统计得到的NGram字典，已排序
                      ngram_max_num=None):          # 要提取前n个NGram，可从统计函数中获取信息，或者不指定

    reporter = Reporter()

    if ngram_dict is not None and ngram_max_num is not None:
        valid_ngrams = list(ngram_dict.keys())[:ngram_max_num]
    else:
        valid_ngrams = None

    for folder in tqdm(os.listdir(parent_path)):
        folder_path = parent_path + folder + '/'

        try:
            ngram_seq = []
            report = loadJson(folder_path + folder + '.json')
            api_seq = report['apis']

            for i in range(len(api_seq) - window):
                ngram = strlistToStr(api_seq[i:i + window])

                # 没有指定要提取的ngram或者当前ngram存在于要提取的ngram中时才会添加
                if valid_ngrams is None or ngram in valid_ngrams:
                    ngram_seq.append(ngram)

            # 写回原文件中
            report['apis'] = ngram_seq
            dumpJson(report, folder_path + folder + '.json')

            reporter.logSuccess()

        except Exception as e:
            reporter.logError(entity=folder, msg=str(e))
            continue

    reporter.report()




