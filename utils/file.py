import json
import os

def loadJson(path):
    with open(path, 'r') as f:
        j = json.load(f)
    return j

def dumpJson(obj, path, indent=4):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)

def dumpIterable(iterable, title, path):
    iter_dict = {title: []}

    for item in iterable:
        iter_dict[title].append(item)

    dumpJson(iter_dict, path)

def deleteDir(path):
    os.system('rm -rf %s'%path)