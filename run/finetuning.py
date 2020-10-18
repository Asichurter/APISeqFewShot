import shutil
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

from models.FT import FT
from models.mconfig import ADAPTED_MODELS, IMP_MODELS
from utils.file import deleteDir, dumpJson

cfg = TrainingConfigManager('ftConfig.json')
datasetBasePath = cfg.systemParams()
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.deviceId())
ft_epoch = cfg.ftEpoch()

sys.setrecursionlimit(5000)                         # 增加栈空间防止意外退出

################################################
#----------------------读取任务参数------------------
################################################

data_folder = cfg.dataset()#'virushare_20_image'

k,n,qk,N = cfg.taskParams()

version = cfg.version()

TestingEpoch = cfg.epoch()
USED_SUB_DATASET = cfg.subDataset()

print('*'*50)
print("Fine-tuning")
print("FineTuning Epoch:", ft_epoch)
print('Used dataset: %s'%data_folder)
print('Version: %d'%version)
print(f"{k}-shot {n}-way")
print(f"device: {cfg.deviceId()}")
print('*'*50)


################################################
#----------------------定义数据------------------
################################################

printState('init managers...')
test_path_manager = PathManager(dataset=data_folder,
                               d_type=USED_SUB_DATASET,
                               version=version)

################################################
#----------------------读取模型参数------------------
################################################

model_cfg = TrainingConfigManager('./runConfig.json')

modelParams = model_cfg.modelParams()

LRDecayIters, LRDecayGamma, optimizer_type, \
weight_decay, loss_func_name, default_lr, lrs, \
taskBatchSize, criteria = model_cfg.trainingParams()

test_dataset = SeqFileDataset(test_path_manager.FileData(),
                               test_path_manager.FileSeqLen(),
                               N)

expand = True if loss_func_name == 'mse' else False


test_task = AdaptEpisodeTask(k, qk, n, N, test_dataset, cuda=True, expand=expand)

stat = TestStatManager(report_cycle=100)

################################################
#----------------------模型定义和初始化------------------
################################################

printState('init model...')
word_matrix = t.Tensor(np.load(test_path_manager.WordEmbedMatrix(), allow_pickle=True))

loss = t.nn.NLLLoss().cuda() if loss_func_name == 'nll' else t.nn.MSELoss().cuda()

model = FT(n=n, loss_fn=loss,
           pretrained_matrix=word_matrix, **modelParams)

# model.load_state_dict(state_dict)
model = model.cuda()

statParamNumber(model)

if os.path.exists(test_path_manager.Doc()):
    deleteDir(test_path_manager.Doc())
os.mkdir(test_path_manager.Doc())
shutil.copy('../models/FT.py', test_path_manager.Doc()+"FT.py")
shutil.copy('./ftConfig.json', test_path_manager.Doc()+"config.json")
print('doc path:', test_path_manager.Doc())

stat.startTimer()

################################################
#--------------------开始测试------------------
################################################

metrics = np.zeros(4,)
with t.autograd.set_detect_anomaly(False):
    for epoch in range(TestingEpoch):
        loss_val = t.zeros((1,)).cuda()
        acc_val = np.zeros(4,) #0.
        model = FT(n=n, loss_fn=loss,
           pretrained_matrix=word_matrix, **modelParams).cuda()
        optimizer = t.optim.SGD(model.parameters(), lr=1e-2)

        (supports, queries, \
        support_len, query_len, \
        support_labels), query_labels = test_task.episode()

        model.train()
        for i in range(ft_epoch):
            sup_preds = model(supports.view(n*k,-1), support_len)
            loss_val = loss(sup_preds, support_labels)
            loss_val.backward()

            optimizer.step()

        model.eval()
        preds = model(queries, query_len)
        loss_val = loss(preds, query_labels)

        acc_val += test_task.metrics(preds, acc_only=False)

        # 记录任务batch的平均正确率和损失值
        stat.record(acc_val[0], loss_val.item(), total_step=TestingEpoch)
        metrics += acc_val

desc = cfg.desc()
desc.append(f"{k}-shot {n}-way")
desc.append('使用%s'%USED_SUB_DATASET)
stat.report(doc_path=test_path_manager.Doc(),
            desc=desc)

metrics /= TestingEpoch
print('Precision:', metrics[1]*100)
print('Recall:', metrics[2]*100)
print('F1-Score:', metrics[3]*100)

t.save(model.state_dict(), test_path_manager.DatasetBase()+f'/models/FT_v{version}.0')

