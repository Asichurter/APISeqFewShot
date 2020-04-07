#######################################################
# preprocessing.py
# 基于API序列的小样本学习

# JSON文件预处理和数据集准备
#######################################################

import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import numpy as np

from utils.error import Reporter
from utils.file import loadJson, dumpJson, dumpIterable
from utils.display import printBulletin, printState
from utils.manager import PathManager

#####################################################
# 本函数用于将Cuckoo报告中的api序列提取出来并存储到json文件中
# 本函数对新式报告和旧式报告都兼容
# 注意：本操作是in-place的
#####################################################
def extractApiFromJson(path):

    reporter = Reporter()

    for i,item_dir in enumerate(os.listdir(path)):
        print(i, item_dir)

        cur_json_path = path + item_dir + '/%s.json'%item_dir

        new_report = {}
        new_report['apis'] = []

        # 此处假设json文件与文件夹同名
        try:
            report = loadJson(cur_json_path)

            # 兼容处理后的报告和未处理的报告
            if 'target' in report:
                new_report['name'] = report['target']['file']['name']
            else:
                new_report['name'] = report['name']

            # 新版本的report，含有api字段
            if 'apis' in report:
                new_report['apis'] = report['apis']

            # 完整的报告中，api位于behavior-processes-calls-api中
            else:
                # 按进程-调用-api的方式逐个收集api调用名称
                api_call_seq = []
                for process in report['behavior']['processes']:
                    for call in process['calls']:
                        api_call_seq.append(call['api'])

                new_report['apis'] = api_call_seq

            reporter.logSuccess()

        # 对于键错误，说明源文件中有错误，应该进行留空处理
        except KeyError as e:
            # name字段已保存，则api留空
            if 'name' in new_report:
                new_report['apis'] = []
                dumpJson(new_report, cur_json_path)

            # 否则直接不处理
            reporter.logError(item_dir, str(e))

        # 其他错误不进行处理
        except Exception as e:
            reporter.logError(item_dir, str(e))

    reporter.report()

#####################################################
# 本函数用于统计Cuckoo报告中的api的长度和所有api，将长度为0
# 或者过短的api调用分别作为错误和警告统计，并绘制分布图。警告和
# 错误可以存储为json文件的形式，还可以使用长度阶梯来查看在某个
# 长度以内的样本比例。
#
# 除了对一个样本一个文件夹的结构适用，也适用于一个类同在一个文
# 件夹的结构，后者需要指定参数 class_dir。
#####################################################
def apiStat(path,
            least_length = 10,          # 最小序列长度
            dump_report_path=None,      # 保存错误和警告报告信息的路径，JSON格式
            dump_apiset_path=None,      # 所有API的集合，JSON格式
            ratio_stairs = [],          # 统计序列长度百分比的阶梯
            class_dir=False):           # 一个文件夹内是单类的所有样本还是单个样本

    reporter = Reporter()

    # 统计api的种类个数
    api_set = set()

    # 统计api的长度
    lengths = []

    # 统计api长度的最大最小值
    min_ = sys.maxsize
    max_ = -1

    for folder in tqdm(os.listdir(path)):

        if class_dir:
            items = os.listdir(path + folder + '/')
            items = list(map(lambda x: x.split('.')[0], items))         # 每个文件都支取其名，不取其扩展名

        else:                               # 如果是单个序列一个文件夹，其名称与文件夹相同
            items = [folder]

        for item in items:
            try:
                # 假定json文件与文件夹同名
                report = loadJson(path + folder + '/%s.json' % item)

                length = len(report['apis'])
                lengths.append(length)

                for api in report['apis']:
                    api_set.add(api)

                # 更新最大最小值
                min_ = min(length, min_)
                max_ = max(length, max_)

                if length == 0:
                    reporter.logError(item, 'api length of 0')
                elif length < least_length:
                    reporter.logWarning(item, 'api length of %d' % length)
                else:
                    reporter.logSuccess()

            except Exception as e:
                reporter.logError(item, str(e))

    printBulletin('Max Length: %d' % max_)
    printBulletin('Min Length: %d' % min_)
    printBulletin('API set（%d in total） %s' % (len(api_set), str(api_set)))

    reporter.report()

    lengths = np.array(lengths)

    for length_stair in ratio_stairs:
        ratio = (lengths < length_stair).sum() / len(lengths)
        printBulletin('Length within %d: %f' % (length_stair, ratio))

    plt.hist(lengths, bins=1000, normed=True, range=(0,10000))
    plt.show()

    if dump_report_path is not None:
        reporter.dump(dump_report_path)

    if dump_apiset_path is not None:
        dumpIterable(api_set, 'api_set', dump_apiset_path)

def statApiFrequency(json_path,
                     is_class_dir=False,
                     threshold=None):

    api_frequency = {}
    total = 0

    for dir_ in tqdm(os.listdir(json_path)):
        dir_path = json_path + dir_ + '/'

        if is_class_dir:
            items = os.listdir(dir_path)
        else:
            items = [dir_+'.json']

        for item in items:
            apis = loadJson(dir_path + item)['apis']

            for api in apis:
                if api not in api_frequency:
                    api_frequency[api] = 0
                api_frequency[api] += 1
                total += 1

    printState('API频率统计')
    # 按照频率降序排列
    api_frequency = sorted(api_frequency.items(), key=lambda x: x[1], reverse=True)

    below_threshold = []

    for i,(api, f) in enumerate(api_frequency):
        print('#%d'%i, api, f/total)
        if threshold is not None:
            # threshold小于1时，定义为频率阈值
            if 1 > threshold > f/total:
                below_threshold.append(api)
            # threshold大于1时，定义为排名阈值
            elif i >= threshold >= 1:
                below_threshold.append(api)

    if threshold is not None:
        printState('低于%f的API(%d个)'%(threshold, len(below_threshold)))
        print(below_threshold)



#####################################################
# 本函数用于将一些非标准的别名或者同一调用的多种类型映射为标准
# 的api名称
#####################################################
def mappingApiNormalize(json_path, mapping, is_class_dir=False):
    reporter = Reporter()

    for folder in tqdm(os.listdir(json_path)):

        items = os.listdir(json_path + folder + '/') if is_class_dir else [folder+'.json']

        for item in items:
            item_path = json_path + folder + '/' + item
            try:
                report = loadJson(item_path)

                for i in range(len(report['apis'])):
                    if report['apis'][i] in mapping:
                        report['apis'][i] = mapping[report['apis'][i]]

                dumpJson(report, item_path)

                reporter.logSuccess()

            except Exception as e:
                reporter.logError(item, str(e))

    reporter.report()


#####################################################
# 本函数用于将原API序列中多个重复值移除冗余为至多2次调用，同时
# 根据指定的选择API来过滤不需要的API，留下指定的API。
#
# 移除冗余操作是在源文件上覆盖的，注意备份
#####################################################
def removeApiRedundance(json_path,
                        selected_apis=None):

    reporter = Reporter()

    for item in tqdm(os.listdir(json_path)):
        item_path = json_path + item + '/%s.json'%item

        try:
            report = loadJson(item_path)

            redun_flag = False
            redun_api_token = None

            new_api_seq = []

            for api_token in report['apis']:
                # 只关注选出的那些api
                # 如果给定的选中API为None代表不进行选择
                if selected_apis is None or \
                    api_token in selected_apis:
                    if api_token != redun_api_token:     # 每当遇到新的api时，刷新当前遇到的api，同时重置flag
                        redun_api_token = api_token
                        redun_flag = False
                    else:
                        if not redun_flag:              # 如果遇到了一样的api，但是flag没有置位，说明第二次遇到，同时置位flag
                            redun_flag = True
                        else:
                            continue                    # 如果遇到了一样的api且flag置位，说明已经遇到过两次，则跳过冗余api

                    new_api_seq.append(api_token)

            # 使用新api序列覆盖原api序列
            report['apis'] = new_api_seq
            dumpJson(report, item_path)

            reporter.logSuccess()

        except Exception as e:
            reporter.logError(item, str(e))

    reporter.report()

def filterApiSequence(json_path,
                      filterd_apis):

    reporter = Reporter()

    for item in tqdm(os.listdir(json_path)):
        item_path = json_path + item + '/%s.json'%item

        try:
            report = loadJson(item_path)

            new_api_seq = []

            for api_token in report['apis']:
                # 只保留不在过滤列表中的API
                if api_token not in filterd_apis:
                    new_api_seq.append(api_token)

            # 使用新api序列覆盖原api序列
            report['apis'] = new_api_seq
            dumpJson(report, item_path)

            reporter.logSuccess()

        except Exception as e:
            reporter.logError(item, str(e))

    reporter.report()


#####################################################
# 本函数用于统计每个类持有的样本数量，会参考已经生成的
# json_w_e_report.json文件中记录的API序列长度过短的文件，
# 无视这些文件进行统计。可以指定阶梯来统计满足大于等于某个值的
# 样本个数，同时将满足条件的类存储为json文件形式。
#####################################################
def statSatifiedClasses(pe_path,
                        json_path,
                        report_path,
                        stat_stairs=[10, 15, 20],
                        count_dump_path=None):
    # 将样本名称映射为类
    cls_mapping = {}
    cls_cnt = {}

    warn_err_report = loadJson(report_path)

    for cls in os.listdir(pe_path):
        cls_cnt[cls] = 0
        for item in os.listdir(pe_path + cls + '/'):
            cls_mapping[item] = cls

    for json_item in os.listdir(json_path):
        if json_item not in warn_err_report['errors'] and \
            json_item not in warn_err_report['warnings']:

            cls_cnt[cls_mapping[json_item]] += 1

    stair_cls_cnt = {}
    for stair in stat_stairs:
        stair_cls_cnt[stair] = []

        for cls_name, cnt in cls_cnt.items():
            if cnt >= stair:
                stair_cls_cnt[stair].append(cls_name)

        printBulletin('More than %d items (%d in total)' % (stair, len(stair_cls_cnt[stair])))

    if count_dump_path is not None:
        dumpJson(stair_cls_cnt, count_dump_path, indent=None)


#####################################################
# 本函数用于根据指定的类，从PE结构中找到对应类的所有样本json，
# 然后将同类的API调用的json全部收集一个文件夹中。指定的类可以
# 根据数量阶梯从stair_cls_cnt.json中提取。
#####################################################
def collectJsonByClass(pe_path, json_path, dst_path, selected_classes):
    reporter = Reporter()

    for cls in tqdm(selected_classes):
        dst_dir = dst_path + cls + '/'

        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        for item in os.listdir(pe_path + cls + '/'):
            try:
                shutil.copy(json_path + item + '/%s.json'%item,
                            dst_dir + '/%s.json'%item)

                reporter.logSuccess()

            except Exception as e:
                reporter.logError('%s/%s' % (cls, item), str(e))

    reporter.report()




if __name__ == '__main__':
    manager = PathManager(dataset='virushare_20', d_type='all')

    '''
    调用顺序：extract -> mapping -> removeRedundance -> (ngram) -> apiStat 
            -> stat_classes -> collect
    '''

    # extractApiFromJson()

    # extractApiFromJson(path)

    # apiStat('D:/peimages/PEs/virushare_20/jsons/',
    #          dump_report_path='D:/peimages/PEs/virushare_20/json_w_e_report.json',
    #          dump_apiset_path='D:/peimages/PEs/virushare_20/api_set.json',
    #         ratio_stairs=[100, 200, 500, 1000, 2000, 3000],
    #         class_dir=False)

    # removeApiRedundance('D:/peimages/PEs/virushare_20/jsons/',
    #                     selected_apis=None)



    # mappingApiNormalize(path,
    #                       mapping={
    #                           "RegCreateKeyExA" : "RegCreateKey",
    #                           "RegCreateKeyExW" : "RegCreateKey",
    #                           "RegDeleteKeyA" : "RegDeleteKey",
    #                           "RegDeleteKeyW" : "RegDeleteKey",
    #                           "RegSetValueExA" : "RegSetValue",
    #                           "RegSetValueExW" : "RegSetValue",
    #                           "RegDeleteValueW" : "RegDeleteValue",
    #                           "RegDeleteValueA" : "RegDeleteValue",
    #                           "RegEnumValueW" : "RegEnumValue",
    #                           "RegEnumValueA" : "RegEnumValue",
    #                           "RegQueryValueExW" : "RegQueryValue",
    #                           "RegQueryValueExA" : "RegQueryValue",
    #                           "CreateProcessInternalW" : "CreateProcess",
    #                           "NtCreateThreadEx" : "NtCreateThread",
    #                           "CreateRemoteThread" : "CreateRemoteThread",
    #                           "CreateThread" : "CreateThread",
    #                           "NtTerminateProcess": "TerminateProcess",
    #                           "NtOpenProcess" : "OpenProcess",
    #                           "InternetOpenUrlA" : "InternetOpenUrl",
    #                           "InternetOpenUrlW" : "InternetOpenUrl",
    #                           "InternetOpenW" : "InternetOpen",
    #                           "InternetOpenA" : "InternetOpen",
    #                           "InternetConnectW" : "InternetConnect",
    #                           "InternetConnectA" : "InternetConnect",
    #                           "HttpOpenRequestW" : "HttpOpenRequest",
    #                           "HttpOpenRequestA" : "HttpOpenRequest",
    #                           "HttpSendRequestA" : "HttpSendRequest",
    #                           "HttpSendRequestW" : "HttpSendRequest",
    #                           "ShellExecuteExW" : "ShellExecute",
    #                           "LdrLoadDll" : "LdrLoadDll",
    #                           "CopyFileW" : "CopyFile",
    #                           "CopyFileA" : "CopyFile",
    #                           "CopyFileExW" : "CopyFile",
    #                           "NtCreateFile" : "CreateFile",
    #                           "DeleteFileW" : "DeleteFile",
    #                           "NtDeleteFile" : "NtDeleteFile",
    #                       })

    # removeApiRedundance('D:/peimages/PEs/virushare_20/jsons/',
    #                       selected_apis=[
    #                           "RegCreateKey",
    #                           "RegDeleteKey",
    #                           "RegSetValue",
    #                           "RegDeleteValue",
    #                           "RegEnumValue",
    #                           "RegQueryValue",
    #                           "CreateProcess",
    #                           "NtCreateThread",
    #                           "CreateRemoteThread",
    #                           "CreateThread",
    #                           "TerminateProcess",
    #                           "OpenProcess",
    #                           "InternetOpenUrl",
    #                           "InternetOpen",
    #                           "InternetConnect",
    #                           "HttpOpenRequest",
    #                           "HttpSendRequest",
    #                           "ShellExecute",
    #                           "LdrLoadDll",
    #                           "CopyFile",
    #                           "CreateFile",
    #                           "DeleteFile",
    #                           "NtDeleteFile"
    #                       ])

    # statSatifiedClasses(pe_path='D:/peimages/PEs/virushare_20/all/',
    #                       json_path='D:/peimages/PEs/virushare_20/jsons/',
    #                       report_path='D:/peimages/PEs/virushare_20/json_w_e_report.json',
    #                       stat_stairs=[20, 15, 10, 5],
    #                       count_dump_path='D:/peimages/PEs/virushare_20/stair_cls_cnt.json')

    # collectJsonByClass(pe_path='D:/peimages/PEs/virushare_20/all/',
    #                       json_path='D:/peimages/PEs/virushare_20/jsons/',
    #                       dst_path='D:/peimages/JSONs/virushare_20/train/',
    #                       selected_classes=["1clickdownload",
    #                                         "4shared",
    #                                         "acda",
    #                                         "adclicer",
    #                                         "airinstaller",
    #                                         "antavmu",
    #                                         "autoit",
    #                                         "badur",
    #                                         "banload",
    #                                         "bettersurf",
    #                                         "black",
    #                                         "blacole",
    #                                         "browsefox",
    #                                         "bundlore",
    #                                         "buterat",
    #                                         "c99shell",
    #                                         "cidox",
    #                                         "conficker",
    #                                         "cpllnk",
    #                                         "darbyen",
    #                                         "darkkomet",
    #                                         "dealply",
    #                                         "decdec",
    #                                         "delbar",
    #                                         "directdownloader",
    #                                         "dlhelper",
    #                                         "domaiq",
    #                                         "downloadadmin",
    #                                         "downloadassistant",
    #                                         "downloadsponsor",
    #                                         "egroupdial",
    #                                         "extenbro",
    #                                         "faceliker",
    #                                         "fakeie",
    #                                         "fbjack",
    #                                         "fearso",
    #                                         "firseria",
    #                                         "fosniw",
    #                                         "fsysna",
    #                                         "fujacks",
    #                                         "gamevance",
    #                                         "gator",
    #                                         "gepys",
    #                                         "getnow",
    #                                         "goredir",
    #                                         "hicrazyk",
    #                                         "hidelink",
    #                                         "hijacker",
    #                                         "hiloti",
    #                                         "ibryte",
    #                                         "icloader",
    #                                         "iframeinject",
    #                                         "iframeref",
    #                                         "includer",
    #                                         "inor",
    #                                         "installerex",
    #                                         "installmonetizer",
    #                                         "instally",
    #                                         "ipamor",
    #                                         "ircbot",
    #                                         "jeefo",
    #                                         "jyfi",
    #                                         "kido",
    #                                         "kovter",
    #                                         "kykymber",
    #                                         "lineage",
    #                                         "linkular",
    #                                         "lipler",
    #                                         "llac",
    #                                         "loadmoney",
    #                                         "loring",
    #                                         "lunam",
    #                                         "mepaow",
    #                                         "microfake",
    #                                         "midia",
    #                                         "mikey",
    #                                         "msposer",
    #                                         "mydoom",
    #                                         "nimda",
    #                                         "nitol",
    #                                         "outbrowse",
    #                                         "patchload",
    #                                         "pirminay",
    #                                         "psyme",
    #                                         "pullupdate",
    #                                         "pykspa",
    #                                         "qhost",
    #                                         "qqpass",
    #                                         "reconyc",
    #                                         "redir",
    #                                         "refresh",
    #                                         "refroso",
    #                                         "scarsi",
    #                                         "scrinject",
    #                                         "sefnit",
    #                                         "shipup",
    #                                         "simbot",
    #                                         "soft32downloader",
    #                                         "softcnapp",
    #                                         "softonic",
    #                                         "softpulse",
    #                                         "somoto",
    #                                         "startp",
    #                                         "staser",
    #                                         "sytro",
    #                                         "toggle",
    #                                         "trymedia",
    #                                         "unruy",
    #                                         "urausy",
    #                                         "urelas",
    #                                         "vilsel",
    #                                         "vittalia",
    #                                         "vtflooder",
    #                                         "wabot",
    #                                         "wajam",
    #                                         "webprefix",
    #                                         "windef",
    #                                         "wonka",
    #                                         "xorer",
    #                                         "xtrat",
    #                                         "yoddos",
    #                                         "zapchast",
    #                                         "zbot",
    #                                         "zegost",
    #                                         "zeroaccess",
    #                                         "zvuzona",
    #                                         "zzinfor"
    #                                     ])

    # apiStat(path=manager.Folder(),
    #         ratio_stairs=[100, 200, 500, 600, 1000, 2000],
    #         class_dir=True)