
import sys
import os

################################################
#----------------------设置系统基本信息------------------
################################################

# appendProjectPath(depth=1)
sys.path.append('../')

# 先添加路径再获取

from components.task import *
from utils.manager import TrainingConfigManager, PathManager, TestStatManager
from components.datasets import SeqFileDataset, ImageFileDataset
from utils.display import printState
from utils.stat import statParamNumber
from components.procedure import *

from models.ProtoNet import ProtoNet
from models.InductionNet import InductionNet
from models.MetaSGD import MetaSGD
from models.ATAML import ATAML
from models.HybridAttentionNet import HAPNet
from models.ConvProtoNet import ConvProtoNet
from models.PerLayerATAML import PerLayerATAML
from models.TCProtoNet import TCProtoNet
from models.AFEAT import AFEAT
from models.FEAT import FEAT
from models.NnNet import NnNet
from models.IMP import IMP
from models.SIMPLE import SIMPLE
from models.HybridIMP import HybridIMP
from models.mconfig import ADAPTED_MODELS, IMP_MODELS

# ADAPTED_MODELS = ['MetaSGD', 'ATAML', 'PerLayerATAML']
# IMP_MODELS = ['IMP', 'SIMPLE', 'HybridIMP']

cfg = TrainingConfigManager('testConfig.json')
datasetBasePath = cfg.systemParams()
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.deviceId())

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
MODEL_RANDOM_STATE = cfg.isRandom()

print('*'*50)
print('Model Name: %s'%model_type)
print('Used dataset: %s'%data_folder)
print('Version: %d'%version)
print(f"{k}-shot {n}-way")
print(f"device: {cfg.deviceId()}")
print("Is Random Test:", MODEL_RANDOM_STATE)
print('*'*50)


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

param_config_path = 'runConfig.json' if MODEL_RANDOM_STATE else test_path_manager.Doc()+'config.json'
model_cfg = TrainingConfigManager(param_config_path)

modelParams = model_cfg.modelParams()

LRDecayIters, LRDecayGamma, optimizer_type,\
weight_decay, loss_func, default_lr, lrs, \
taskBatchSize, criteria = model_cfg.trainingParams()

test_dataset = SeqFileDataset(test_path_manager.FileData(),
                               test_path_manager.FileSeqLen(),
                               N)

expand = True if loss_func=='mse' else False

if model_type in ADAPTED_MODELS:
    test_task = AdaptEpisodeTask(k, qk, n, N, test_dataset,
                                  cuda=True, expand=expand)
elif model_type in IMP_MODELS:
    test_task = ImpEpisodeTask(k, qk, n, N, test_dataset,
                                  cuda=True, expand=expand)
else:
    test_task = ProtoEpisodeTask(k, qk, n, N, test_dataset,
                                  cuda=True, expand=expand)

stat = TestStatManager()

################################################
#----------------------模型定义和初始化------------------
################################################

printState('init model...')
if not MODEL_RANDOM_STATE:
    state_dict = t.load(test_path_manager.Model(type=cfg.loadBest()))
    if model_type in ADAPTED_MODELS:
        word_matrix = state_dict['Learner.Embedding.weight']
    else:
        word_matrix = state_dict['Embedding.weight']
else:
    word_matrix = t.Tensor(np.load(test_path_manager.WordEmbedMatrix(), allow_pickle=True))
print("loading done...")

loss = t.nn.NLLLoss().cuda() if loss_func=='nll' else t.nn.MSELoss().cuda()

if model_type == 'ProtoNet':
    model = ProtoNet(pretrained_matrix=word_matrix,
                     **modelParams)
elif model_type == 'InductionNet':
    model = InductionNet(pretrained_matrix=word_matrix,
                         **modelParams)
elif model_type == 'MetaSGD':
    model = MetaSGD(n=n,
                    loss_fn=loss,
                    pretrained_matrix=word_matrix,
                    **modelParams
                    )
elif model_type == 'ATAML':
    model = ATAML(n=n,
                  loss_fn=loss,
                  pretrained_matrix=word_matrix,
                  **modelParams
                  )
elif model_type == 'HybridAttentionNet':
    model = HAPNet(k=k,
                   pretrained_matrix=word_matrix,
                   **modelParams
                   )
elif model_type == 'ConvProtoNet':
    model = ConvProtoNet(k=k,
                   pretrained_matrix=word_matrix,
                   **modelParams
                   )
elif model_type == 'PerLayerATAML':
    model = PerLayerATAML(n=n,
                          loss_fn=loss,
                          pretrained_matrix=word_matrix,
                          **modelParams
                          )
# elif model_type == 'Reptile':
#     model = Reptile(n=n,
#                     loss_fn=loss,
#                     pretrained_matrix=word_matrix,
#                     **modelParams
#                     )
elif model_type == 'TCProtoNet':
    model = TCProtoNet(pretrained_matrix=word_matrix,
                        **modelParams)
elif model_type == 'FEAT':
    model = FEAT(pretrained_matrix=word_matrix,
                 **modelParams)
elif model_type == 'AFEAT':
    model = AFEAT(pretrained_matrix=word_matrix,
                 **modelParams)
elif model_type == 'NnNet':
    model = NnNet(pretrained_matrix=word_matrix,
                  **modelParams)
elif model_type == 'IMP':
    model = IMP(pretrained_matrix=word_matrix,
                     **modelParams)
elif model_type == 'SIMPLE':
    model = SIMPLE(pretrained_matrix=word_matrix,
                   **modelParams)
elif model_type == 'HybridIMP':
    model = HybridIMP(pretrained_matrix=word_matrix,
                      **modelParams)

if not MODEL_RANDOM_STATE:
    model.load_state_dict(state_dict)
model = model.cuda()

statParamNumber(model)

stat.startTimer()

################################################
#--------------------开始测试------------------
################################################
metrics = np.zeros(4,)
with t.autograd.set_detect_anomaly(False):
    for epoch in range(TestingEpoch):

        if model_type == 'TCProtoNet':
            acc_val, loss_val_item =penalQLossProcedure(model,
                                                        1,
                                                        test_task,
                                                        loss,
                                                        None,
                                                        None,
                                                        train=False)

        elif model_type == 'FEAT':
            acc_val, loss_val_item = featProcedure(model,
                                                 n,k,qk,
                                                 1,
                                                 test_task,
                                                 loss,
                                                 None,
                                                 None,
                                                 train=False,
                                                 contrastive_factor=modelParams['contrastive_factor'])

        elif model_type in IMP_MODELS :
            acc_val, loss_val_item = impProcedure(model,
                                                  1,
                                                  test_task,
                                                  None,
                                                  None,
                                                  train=False,
                                                  acc_only=False)

        else:
            acc_val, loss_val_item = queryLossProcedure(model,
                                                        1,
                                                        test_task,
                                                        loss,
                                                        None,
                                                        None,
                                                        train=False,
                                                        acc_only=False)

        # 记录任务batch的平均正确率和损失值
        stat.record(acc_val[0], loss_val_item, total_step=TestingEpoch)
        metrics += acc_val

desc = cfg.desc()
desc.append(f"{k}-shot {n}-way")
desc.append('使用%s'%USED_SUB_DATASET)
stat.report(doc_path=None if MODEL_RANDOM_STATE else test_path_manager.Doc(),
            desc=desc)

metrics /= TestingEpoch
print('Precision:', metrics[1]*100)
print('Recall:', metrics[2]*100)
print('F1-Score:', metrics[3]*100)

