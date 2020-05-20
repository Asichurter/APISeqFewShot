#########################################################
# This script is used to delete useless directory in the
# analyses directory to release space for further analyses.
# This script can be executed when low space warning is
# raised.
#########################################################

import os
from tqdm import tqdm

exception_dir = ['latest']
del_child_dirs = ['buffer', 'extracted', 'files', 'memory', 'network', 'logs', 'shots']

shell_temp = 'rm -rf {path}'

analyses_path = '/home/asichurter/datasets/JSONs/LargePE-Unfiltered-original/'

for d in tqdm(os.listdir(analyses_path)):
    if d not in exception_dir:
        # print(d)
        for del_dir in del_child_dirs:
            if os.path.exists(analyses_path + d + '/%s/'%del_dir):
                ret = os.system(shell_temp.format(path=analyses_path + d + '/%s/'%del_dir))
                if ret != 0:
                    print('\n\nFail to execute shell script for %s!\n\n'%d)
                # break
