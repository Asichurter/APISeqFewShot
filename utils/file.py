import json

def loadJson(path):
    with open(path, 'r') as f:
        j = json.load(f)
    return j

def dumpJson(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def dumpIterable(iterable, title, path):
    iter_dict = {title: []}

    for item in iterable:
        iter_dict[title].append(item)

    with open(path, 'w') as f:
        json.dump(iter_dict, f)