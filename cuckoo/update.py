#########################################################
# This script aims to resolve the completed and reported
# malware which has json in storage and refresh the json 
# config in the malware dataset directory.
#
# Once failure occurs when running, try to run this script
# to make json config in dataset up to date.
#########################################################


import json
import os
import logging

path = "/home/asichurter/.cuckoo/storage/analyses/"
# path = "/home/asichurter/datasets/JSONs/LargePE-100-original/no-problem/"
child_path = '/reports/report.json'
config_path = "/home/asichurter/datasets/PEs/LargePE-100/config.json"
# hist_config_path = "/home/asichurter/malwares/virushare_20/histConfig.json"

with open(config_path, 'r') as f:
    cfgs = json.load(f)

success_cnt = 0
errors = []
warnings = []
current_cfgs = {}

unused_list = ['latest']

for i, task_id in enumerate(os.listdir(path)):
    if task_id in unused_list:
        continue

    if os.path.isdir(path + task_id + '/'):
        try:
            print('------------------------------------------------')
            print(str(i) + ' ' +
                  task_id + ': ' +
                  str(float(os.path.getsize(path + task_id + child_path)) / (1024.0 ** 2)) + ' MB')  #
            with open(path + task_id + '/' + child_path, 'r') as json_f:  #
                report = json.load(json_f)
                file_name = report['target']['file']['name']

                print('api length: ' + str(len(report['api'])))
                print('------------------------------------------------')

                if len(report['api']) < 10:
                    warnings.append('task_id=%s has length of %d'%(task_id, len(report['api'])))

                # identical file appears twice in the pass, ignore and warn
                if file_name in current_cfgs:
                    warnings.append('task_id=%s appears twice in the pass, last appear in task_id=%s, file=%s'%
                                        (task_id, current_cfgs[file_name], file_name))
                    
                # if the record has been in the cfgs and has different task_id
                # it means the same file appears twice in the history
                elif file_name in cfgs:
                    if cfgs[file_name] != int(task_id):
                        warnings.append('task_id=%s has already in the cfgs'%task_id)

                else:
                    cfgs[file_name] = int(task_id)

            success_cnt += 1
            current_cfgs[file_name] = int(task_id)

        except Exception as e:
            errors.append('task_id=%s error:%s' % (task_id, str(e)))

with open(config_path, 'w') as f:
    json.dump(cfgs, f, indent=4)

print('\n\n------------------%d errors----------------------------'%len(errors))
for e in errors:
    print(e)
print('-----------------------------------------------------\n\n')

print('\n\n------------------%d warnings----------------------------'%len(warnings))
for w in warnings:
    print(w)
print('-----------------------------------------------------\n\n')

print('Successfully parsed %d analysis report' % success_cnt)
