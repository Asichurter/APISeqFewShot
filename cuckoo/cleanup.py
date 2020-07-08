#########################################################
# This script is used to delete useless directory in the
# analyses directory to release space for further analyses.
# This script can be executed when low space warning is
# raised.
#########################################################

import os
from tqdm import tqdm

exception_dir = ['latest']

rmv_temp = 'rm -rf {path}'
mv_temp = 'mv {path}/reports/report.json {path}/report.json && rm -rf {path}/reports'

analyses_path = '/home/asichurter/datasets/JSONs/virushare-20-3gram-incre/'
# analyses_path = '/home/asichurter/codes/test/'

rename = True

for d in tqdm(os.listdir(analyses_path)):
    if d not in exception_dir:
        # print(d)
        for item in os.listdir(analyses_path+d):
            if os.path.isdir(analyses_path+d+'/'+item):    
                if item != 'reports':
                    os.system(rmv_temp.format(path=analyses_path+d+'/%s/'%item))
            else:
                os.remove(analyses_path+d+'/'+item)
        
        os.system(mv_temp.format(path=analyses_path+d))
