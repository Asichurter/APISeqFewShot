#########################################################
# This script is used to submit all samples in the dataset
# to analyze. All the candidate samples will be checked
# according to the config file and existed samples will
# not be submitted again.

# When using this script to resubmit missing samples, be
# sure to run "update" before running this script to ensure
# the validity.

# Note: This script should better run in Python 2 environment
# where Werkzeug==0.16.1.
#########################################################


import requests
import os
import shutil
import json

pe_path = '/home/asichurter/datasets/PEs/cache/'
config_path = '/home/asichurter/datasets/PEs/LargePE-Unfiltered/config.json'

port = 1337
token = 'kbU-z_aDD3CT4RCPE9ayMg'

REST_URL = "http://localhost:{port}/tasks/create/file".format(port=port)
HEADERS = {"Authorization": "Bearer " + token}

erros = []

# read completed sample name and corresponding ID
with open(config_path, 'r') as f:
    compl_samples = json.load(f)

# with open(config_path, 'w') as f:
for c_i, c in enumerate(os.listdir(pe_path)):
    for i_i, item in enumerate(os.listdir(pe_path + c + '/')):
        print(c_i, i_i)
        if item in compl_samples:
            continue
        with open(pe_path + c + '/' + item, "rb") as sample:
            files = {"file": (item, sample)}
            try:
                r = requests.post(REST_URL, headers=HEADERS, files=files)
                # print(r)
                task_id = r.json()["task_id"]
                if task_id is None:
                    print('item %s in %s has numbered %d has None for task_id' %
                          (item, c, i_i))
                    # raise Exception('task_id is None for %s'%item)
                compl_samples[item] = task_id
                # print(r)
            except Exception as e:
                print(e)
                # json.dump(compl_samples, f)
                exit(-1)



