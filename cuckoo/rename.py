#########################################################
# This script is to rename the unconsistent naming scheme 
# of 'API' field.
#########################################################

import os
import json
from tqdm import tqdm

json_path = '/home/asichurter/datasets/JSONs/LargePE-100-original/'

for folder in tqdm(os.listdir(json_path)):
    with open(json_path+folder+'/'+folder+'.json', 'r') as f:
        report = json.load(f)

    if 'apis' not in report:
        apis = report['api']
        report['apis'] = apis
        del report['api']
    
    with open(json_path+folder+'/'+folder+'.json', 'w') as f:
        json.dump(report, f, indent=4)
