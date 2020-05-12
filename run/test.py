
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

from models.ProtoNet import ProtoNet, ImageProtoNet, IncepProtoNet, CNNLstmProtoNet
from models.InductionNet import InductionNet
from models.MetaSGD import MetaSGD
from models.ATAML import ATAML
from models.HybridAttentionNet import HAPNet
from models.ConvProtoNet import ConvProtoNet
from models.PreLayerATAML import PreLayerATAML

ADAPTED_MODELS = ['MetaSGD', 'ATAML', 'PreLayerATAML']

cfg = TrainingConfigManager('testConfig.json')
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

EmbedSize, HiddenSize, BiLstmLayer, SelfAttDim, usePretrained,\
wordCnt = model_cfg.modelParams()

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

if model_type == 'ProtoNet':
    model = ProtoNet(pretrained_matrix=word_matrix,
                     embed_size=EmbedSize,
                     hidden=HiddenSize,
                     layer_num=BiLstmLayer,
                     self_attention=SelfAttDim is not None,
                     self_att_dim=SelfAttDim,
                     word_cnt=wordCnt)
elif model_type == 'InductionNet':
    model = InductionNet(pretrained_matrix=word_matrix,
                         embed_size=EmbedSize,
                         hidden_size=HiddenSize,
                         layer_num=BiLstmLayer,
                         self_att_dim=SelfAttDim,
                         ntn_hidden=100, routing_iters=3,
                         word_cnt=wordCnt,
                         freeze_embedding=False)
elif model_type == 'MetaSGD':
    model = MetaSGD(n=n,
                    loss_fn=loss,
                    pretrained_matrix=word_matrix,
                    embed_size=EmbedSize
                    # hidden_size=HiddenSize,
                    # layer_num=BiLstmLayer,
                    # self_att_dim=SelfAttDim
                    # word_cnt=wordCnt,
                    # freeze_embedding=False
                    )
elif model_type == 'ATAML':
    model = ATAML(n=n,
                  loss_fn=loss,
                  pretrained_matrix=word_matrix,
                  embed_size=EmbedSize,
                  hidden_size=HiddenSize,
                  layer_num=BiLstmLayer,
                  self_att_dim=SelfAttDim
                  )
elif model_type == 'HybridAttentionNet':
    model = HAPNet(k=k,
                   pretrained_matrix=word_matrix,
                   embed_size=EmbedSize,
                   hidden_size=HiddenSize,
                   layer_num=BiLstmLayer,
                   self_att_dim=SelfAttDim,
                   word_cnt=wordCnt
                   )
elif model_type == 'ConvProtoNet':
    model = ConvProtoNet(k=k,
                   pretrained_matrix=word_matrix,
                   embed_size=EmbedSize,
                   hidden_size=HiddenSize,
                   layer_num=BiLstmLayer,
                   self_att_dim=SelfAttDim,
                   word_cnt=wordCnt
                   )

model.load_state_dict(state_dict)
model = model.cuda()

statParamNumber(model)

stat.startTimer()

################################################
#--------------------开始测试------------------
################################################
with t.autograd.set_detect_anomaly(False):
    for epoch in range(TestingEpoch):
        model.eval()

        loss_val = t.zeros((1,)).cuda()
        acc_val = 0.

        model_input, labels = test_task.episode()

        predicts = model(*model_input)

        loss_val += loss(predicts, labels)

        predicts = predicts.cpu()
        acc_val = test_task.accuracy(predicts)
        loss_val_item = loss_val.detach().item()

        # 记录任务batch的平均正确率和损失值
        stat.record(acc_val, loss_val_item)

desc = cfg.desc()
desc.append('使用%s'%USED_SUB_DATASET)
stat.report(doc_path=test_path_manager.Doc(),
            desc=desc)

