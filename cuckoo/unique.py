#########################################################
# This script is used to delete the duplicated sample 
# reports of identical samples.
#########################################################

import os
import json
from tqdm import tqdm

rmv_temp = 'rm -rf {path}'

json_path = '/home/asichurter/datasets/JSONs/LargePE-100-original/'

item_list = []

for folder in tqdm(os.listdir(json_path)):
    with open(json_path+folder+'/reports/report.json') as f:
        report = json.load(f)
        name = report['target']['file']['name']
        if name in item_list:
            os.system(rmv_temp.format(path=json_path+folder))
        else:
            item_list.append(name)

print(len(item_list), 'unique samples in total.')