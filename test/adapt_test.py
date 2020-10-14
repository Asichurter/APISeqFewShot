import numpy as np
from tqdm import tqdm
import sys
from config import appendProjectPath, saveConfigFile

################################################
#----------------------设置系统基本信息------------------
################################################

appendProjectPath(depth=1)

# 先添加路径再获取

from components.task import *
from utils.manager import TrainingConfigManager, PathManager, TestStatManager
from components.datasets import SeqFileDataset, ImageFileDataset
from utils.display import printState
from utils.stat import statParamNumber
from components.procedure import *

from models.FEAT import FEAT

ADAPTED_MODELS = ['MetaSGD', 'ATAML', 'PerLayerATAML']

cfg = TrainingConfigManager('../run/testConfig.json')
datasetBasePath = cfg.systemParams()

sys.setrecursionlimit(5000)                         # 增加栈空间防止意外退出

################################################
#----------------------读取任务参数------------------
################################################

data_folder = cfg.dataset()#'virushare_20_image'

k,n,qk,N = cfg.taskParams()
model_type, model_name = cfg.model()

version = cfg.version()

TestingEpoch = cfg.epoch()
USED_SUB_DATASET = cfg.subDataset()


################################################
#----------------------定义数据------------------
################################################

printState('init managers...')
test_path_manager = PathManager(dataset=data_folder,
                               d_type=USED_SUB_DATASET,
                               model_name=model_name,
                               version=version)

################################################
#----------------------读取模型参数------------------
################################################

model_cfg = TrainingConfigManager(test_path_manager.Doc()+'config.json')

modelParams = model_cfg.modelParams()

LRDecayIters, LRDecayGamma, optimizer_type,\
weight_decay, loss_func, default_lr, lrs, taskBatchSize = model_cfg.trainingParams()

test_dataset = SeqFileDataset(test_path_manager.FileData(),
                               test_path_manager.FileSeqLen(),
                               N)

expand = True if loss_func=='mse' else False

if model_type in ADAPTED_MODELS:
    test_task = AdaptEpisodeTask(k, qk, n, N, test_dataset,
                                  cuda=True, expand=expand)
else:
    test_task = ProtoEpisodeTask(k, qk, n, N, test_dataset,
                                  cuda=True, expand=expand)

stat = TestStatManager()

################################################
#----------------------模型定义和初始化------------------
################################################

printState('init model...')
state_dict = t.load(test_path_manager.Model())


if model_type in ADAPTED_MODELS:
    word_matrix = state_dict['Learner.Embedding.weight']
else:
    word_matrix = state_dict['Embedding.weight']

loss = t.nn.NLLLoss().cuda() if loss_func=='nll' else t.nn.MSELoss().cuda()

if model_type == 'FEAT':
    model = FEAT(pretrained_matrix=word_matrix,
                 **modelParams)
else:
    raise ValueError('不支持的模型类型:',model_type)


model.load_state_dict(state_dict)
model = model.cuda()

statParamNumber(model)

stat.startTimer()

################################################
#--------------------开始测试------------------
################################################
with t.autograd.set_detect_anomaly(False):
    bef_adapt_acc = []
    bef_adapt_los = []
    aft_adapt_acc = []
    aft_adapt_los = []

    for epoch in range(TestingEpoch):
        (bef_acc,bef_los),(aft_acc,aft_los) = adaFeatProcedure(model,
                                                             n,k,qk,
                                                             test_task,
                                                             loss)
        bef_adapt_acc.append(bef_acc)
        bef_adapt_los.append(bef_los)
        aft_adapt_acc.append(aft_acc)
        aft_adapt_los.append(aft_los)

        if (epoch+1)%100 == 0:
            print('*' * 50)
            print('Before adapted acc: %.4f' % np.mean(bef_adapt_acc[epoch-99:epoch+1]))
            print('After adapted acc: %.4f' % np.mean(aft_adapt_acc[epoch-99:epoch+1]))
            print('*' * 50)
            print('Before adapted loss: %.4f' % np.mean(bef_adapt_los[epoch-99:epoch+1]))
            print('After adapted loss: %.4f' % np.mean(aft_adapt_los[epoch-99:epoch+1]))
            print('*' * 50)
            print('')

print('Final Statistics:')
print('*'*50)
print('Before adapted acc: %.4f'%np.mean(bef_adapt_acc))
print('After adapted acc: %.4f'%np.mean(aft_adapt_acc))
print('*'*50)
print('Before adapted loss: %.4f'%np.mean(bef_adapt_los))
print('After adapted loss: %.4f'%np.mean(aft_adapt_los))
print('*'*50)
print('')


