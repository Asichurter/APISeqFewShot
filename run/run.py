import os
import sys
import shutil

sys.path.append('../')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config import saveConfigFile, checkVersion

################################################
#----------------------设置系统基本信息------------------
################################################
# appendProjectPath(depth=1)


# 先添加路径再获取
from utils.manager import TrainingConfigManager

cfg = TrainingConfigManager('runConfig.json')
datasetBasePath = cfg.systemParams()

sys.setrecursionlimit(5000)                         # 增加栈空间防止意外退出

import torch as t
import numpy as np

from components.task import *
from utils.manager import PathManager, TrainStatManager
from utils.plot import VisdomPlot, plotLine
from components.datasets import SeqFileDataset, ImageFileDataset
from utils.init import LstmInit
from utils.display import printState
from utils.stat import statParamNumber
from utils.file import deleteDir
from components.procedure import *

from models.ProtoNet import ProtoNet, ImageProtoNet, IncepProtoNet, CNNLstmProtoNet
from models.InductionNet import InductionNet
from models.MetaSGD import MetaSGD
from models.ATAML import ATAML
from models.HybridAttentionNet import HAPNet
from models.ConvProtoNet import ConvProtoNet
from models.PerLayerATAML import PerLayerATAML
from models.Reptile import Reptile
from models.TCProtoNet import TCProtoNet
from models.FEAT import FEAT
from models.AFEAT import AFEAT
from models.MatchNet import MatchNet
from models.IMP import IMP
from models.ImpIMP import ImpIMP
from models.HybridIMP import HybridIMP

################################################
#----------------------读取参数------------------
################################################

ADAPTED_MODELS = ['MetaSGD', 'ATAML', 'PerLayerATAML']
IMP_MODELS = ['IMP', 'ImpIMP', 'HybridIMP']

data_folder = cfg.dataset()#'virushare_20_image'

k,n,qk,N = cfg.taskParams()

LRDecayIters,\
LRDecayGamma,\
optimizer_type,\
weight_decay, \
loss_func,\
default_lr,\
lrs, \
taskBatchSize, criteria = cfg.trainingParams()

modelParams = cfg.modelParams()

model_type, model_name = cfg.model()

version = cfg.version()

ValCycle,\
ValEpisode = cfg.valParams()

RecordGradient,\
GradientUpdateCycle = cfg.gradRecParams()

TrainingVerbose,\
UseVisdom = cfg.verboseParams()

types,\
titles,\
xlabels,\
ylabels,\
legends = cfg.plotParams()

TrainingEpoch = cfg.epoch()

print('*'*50)
print('Model Name: %s'%model_type)
print('Used dataset: %s'%data_folder)
print('Version: %d'%version)
print('*'*50)

################################################
#----------------------定义数据------------------
################################################

expand = False if loss_func=='nll' else True

loss = t.nn.NLLLoss().cuda() \
    if loss_func=='nll' else \
    t.nn.MSELoss().cuda()

printState('init managers...')
train_path_manager = PathManager(dataset=data_folder,
                                 d_type='train',
                                 model_name=model_name,
                                 version=version)
val_path_manager = PathManager(dataset=data_folder,
                               d_type='validate',
                               model_name=model_name,
                               version=version)

train_dataset = SeqFileDataset(train_path_manager.FileData(),
                               train_path_manager.FileSeqLen(),
                               N)
val_dataset = SeqFileDataset(val_path_manager.FileData(),
                               val_path_manager.FileSeqLen(),
                               N)
# train_dataset = ImageFileDataset(train_path_manager.FileData(), N, rd_crop_size=224)
# val_dataset = ImageFileDataset(val_path_manager.FileData(), N, rd_crop_size=224)

# train_task = MatrixProtoEpisodeTask(k ,qk, n, N,
#                         dataset=train_dataset,
#                         cuda=True,
#                         label_expand=expand,
#                         unsqueeze=False)
# val_task = MatrixProtoEpisodeTask(k ,qk, n, N,
#                         dataset=val_dataset,
#                         cuda=True,
#                         label_expand=expand,
#                         unsqueeze=False)

if model_type in ADAPTED_MODELS:
    train_task = AdaptEpisodeTask(k, qk, n, N, train_dataset,
                                  cuda=True, expand=expand,
                                  parallel=modelParams['data_parallel_devices'])
    val_task = AdaptEpisodeTask(k, qk, n, N, val_dataset,
                                  cuda=True, expand=expand,
                                parallel=modelParams['data_parallel_devices'])
elif model_type == 'Reptile':
    train_task = ReptileEpisodeTask(N, n, N,
                                    dataset=train_dataset,
                                    expand=expand,
                                    parallel=modelParams['data_parallel_devices'])
    val_task = ReptileEpisodeTask(N-k, n, N,
                                    dataset=val_dataset,
                                    expand=expand,
                                  parallel=modelParams['data_parallel_devices'])
elif model_type in IMP_MODELS:
    train_task = ImpEpisodeTask(k, qk, n, N, train_dataset,
                                  cuda=True, expand=expand,
                                parallel=modelParams['data_parallel_devices'])
    val_task = ImpEpisodeTask(k, qk, n, N, val_dataset,
                                  cuda=True, expand=expand,
                              parallel=modelParams['data_parallel_devices'])

else:
    train_task = ProtoEpisodeTask(k, qk, n, N, train_dataset,
                                  cuda=True, expand=expand,
                                  parallel=modelParams['data_parallel_devices'])
    val_task = ProtoEpisodeTask(k, qk, n, N, val_dataset,
                                  cuda=True, expand=expand,
                                parallel=modelParams['data_parallel_devices'])

stat = TrainStatManager(model_save_path=train_path_manager.Model(),
                        stat_save_path=train_path_manager.Doc(),
                        train_report_iter=ValCycle,
                        criteria=criteria)

if RecordGradient:
    types.append('line')
    titles.append('gradient')
    xlabels.append('iterations')
    ylabels.append('gradient norm')
    legends.append(['Encoder Gradient'])

if UseVisdom:
    plot = VisdomPlot(env_title='train monitoring',
                      types=types,
                      titles=titles,
                      xlabels=xlabels,
                      ylabels=ylabels,
                      legends=legends)

if modelParams['usePretrained']:
    word_matrix = t.Tensor(np.load(train_path_manager.WordEmbedMatrix(), allow_pickle=True))
else:
    word_matrix = None



################################################
#----------------------模型定义和初始化------------------
################################################

printState('init model...')

# model = CNNLstmProtoNet()
# model = IncepProtoNet(channels=[1, 32, 1],
#                       depth=3)
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
elif model_type == 'Reptile':
    model = Reptile(n=n,
                    loss_fn=loss,
                    pretrained_matrix=word_matrix,
                    **modelParams
                    )
elif model_type == 'TCProtoNet':
    model = TCProtoNet(pretrained_matrix=word_matrix,
                        **modelParams)
elif model_type == 'FEAT':
    model = FEAT(pretrained_matrix=word_matrix,
                 **modelParams)
elif model_type == 'AFEAT':
    model = AFEAT(pretrained_matrix=word_matrix,
                 **modelParams)
elif model_type == 'MatchNet':
    model = MatchNet(pretrained_matrix=word_matrix,
                     **modelParams)
elif model_type == 'IMP':
    model = IMP(pretrained_matrix=word_matrix,
                     **modelParams)
elif model_type == 'ImpIMP':
    model = ImpIMP(pretrained_matrix=word_matrix,
                     **modelParams)
elif model_type == 'HybridIMP':
    model = HybridIMP(pretrained_matrix=word_matrix,
                      **modelParams)
# model = ImageProtoNet(in_channels=1)

model = model.cuda()

statParamNumber(model)

# 模型初始化
printState('init parameters...')
# model.apply(LstmInit)

parameters = []
for name, par in model.named_parameters():
    # print(name)
    if name in lrs:
        parameters += [{'params': [par], 'lr':lrs[name], 'weight_decay': weight_decay}]
    else:
        parameters += [{'params': [par], 'lr':default_lr, 'weight_decay': weight_decay}]

from torch.optim.rmsprop import RMSprop
if optimizer_type == 'adam':
    # optimizer = t.optim.AdamW(model.parameters(), lr=default_lr, weight_decay=weight_decay)
    optimizer = t.optim.AdamW(parameters)
elif optimizer_type == 'adagrad':
    optimizer = t.optim.Adagrad(model.parameters(), lr=default_lr, weight_decay=weight_decay)
elif optimizer_type == 'rmsprop':
    optimizer = t.optim.RMSprop(model.parameters(), lr=default_lr, weight_decay=weight_decay)
    # optimizer = RMSprop(parameters, momentum=0.9)
elif optimizer_type == 'sgd':
    optimizer = t.optim.SGD(parameters,momentum=0.9,weight_decay=weight_decay)
else:
    raise ValueError

scheduler = t.optim.lr_scheduler.StepLR(optimizer,
                                        step_size=LRDecayIters,
                                         gamma=LRDecayGamma)


if modelParams['data_parallel']:
    model = t.nn.DataParallel(model,
                              device_ids=modelParams["data_parallel_devices"])

################################################
#----------------------训练------------------
################################################


grad = 0.

printState('start training')
stat.startTimer()

# 检查版本号，以防止不小心覆盖version
# checkVersion(version)
# 保存配置文件到doc
saveConfigFile(train_path_manager.Doc(), model_type)

with t.autograd.set_detect_anomaly(False):
    for epoch in range(TrainingEpoch):
        # print('Epoch', epoch)
        if TrainingVerbose:
            printState('Epoch %d'%epoch)

        if model_type == 'TCProtoNet':
            acc_val, loss_val_item =penalQLossProcedure(model,
                                                        taskBatchSize,
                                                        train_task,
                                                        loss,
                                                        optimizer,
                                                        scheduler,
                                                        train=True)

        elif model_type == 'FEAT':
            acc_val, loss_val_item = featProcedure(model,
                                                 n,k,qk,
                                                 taskBatchSize,
                                                 train_task,
                                                 loss,
                                                 optimizer,
                                                 scheduler,
                                                 train=True,
                                                 contrastive_factor=modelParams['contrastive_factor'])

        elif model_type in IMP_MODELS:
            acc_val, loss_val_item = impProcedure(model,
                                                  taskBatchSize,
                                                  train_task,
                                                  optimizer,
                                                  scheduler,
                                                  train=True)

        else:

            # acc_val, loss_val_item = fomamlProcedure(model,
            #                                          taskBatchSize,
            #                                          train_task,
            #                                          loss,
            #                                          optimizer,
            #                                          scheduler,
            #                                          train=True)

            acc_val, loss_val_item = queryLossProcedure(model,
                                                        taskBatchSize,
                                                        train_task,
                                                        loss,
                                                        optimizer,
                                                        scheduler,
                                                        train=True)

            # acc_val, loss_val_item = reptileProcedure(n, k, model,
            #                                           taskBatchSize=None,
            #                                           task=train_task,
            #                                           loss=loss,
            #                                           train=True)

        if RecordGradient:
            grad = 0.
            # 监视Encoder的梯度
            for weight_list in model.Encoder.Encoder.all_weights:
                for w in weight_list:
                    grad += t.norm(w.grad.detach()).item()
            if UseVisdom:
                plot.update('gradient', epoch, [[grad]],
                            update={'flag': True,
                                    'val': None if epoch%GradientUpdateCycle==0 else 'append'})

        # 记录任务batch的平均正确率和损失值
        stat.recordTraining(acc_val / taskBatchSize,
                            loss_val_item/ taskBatchSize)


        ################################################
        # ----------------------验证------------------
        ################################################

        if epoch % ValCycle == 0:
            printState('Test in Epoch %d'%epoch)
            # model.eval()
            validate_acc = 0.
            validate_loss = 0.
            
            for test_i in range(ValEpisode):
            #
            # for i in range(ValEpisode):
            #     model_input, labels =val_task.episode()#support, query, sup_len, que_len, labels = val_task.episode()
            #     # support, query, labels = val_task.episode()
            #
            #     # print(model_input[0].size(), model_input[1].size())
            #     predicts = model(*model_input)
            #
            #     loss_val = loss(predicts, labels)
            #
            #     predicts = predicts.cpu()
            #     validate_loss += loss_val.detach().item()
            #     validate_acc += val_task.accuracy(predicts)

                if model_type == 'TCProtoNet':
                    validate_acc_oneiter, validate_loss_oneiter = penalQLossProcedure(model,
                                                                 1,
                                                                 val_task,
                                                                 loss,
                                                                 None,
                                                                 None,
                                                                 train=False)
                    validate_acc += validate_acc_oneiter
                    validate_loss += validate_loss_oneiter
    
                elif model_type == 'FEAT':
                    validate_acc_oneiter, validate_loss_oneiter = featProcedure(model,
                                                           n, k, qk,
                                                           1,
                                                           val_task,
                                                           loss,
                                                           None,
                                                           None,
                                                           train=False,
                                                           contrastive_factor=modelParams['contrastive_factor'])
                    validate_acc += validate_acc_oneiter
                    validate_loss += validate_loss_oneiter
    
                elif model_type in IMP_MODELS:
                    validate_acc_oneiter, validate_loss_oneiter = impProcedure(model,
                                                          1,
                                                          val_task,
                                                          None,
                                                          None,
                                                          train=False)
                    validate_acc += validate_acc_oneiter
                    validate_loss += validate_loss_oneiter
    
                else:
    
                    validate_acc_oneiter, validate_loss_oneiter = queryLossProcedure(model,
                                                                1,
                                                                val_task,
                                                                loss,
                                                                None,
                                                                None,
                                                                train=False)
                    validate_acc += validate_acc_oneiter
                    validate_loss += validate_loss_oneiter

            avg_validate_acc = validate_acc / ValEpisode
            avg_validate_loss = validate_loss / ValEpisode

            stat.recordValidating(avg_validate_acc,
                                  avg_validate_loss,
                                  model,
                                  epoch,
                                  TrainingEpoch)

            train_acc, train_loss, val_acc, val_loss = stat.getRecentRecord()
            if UseVisdom:
                plot.update(title='accuracy', x_val=epoch, y_val=[[train_acc, val_acc]])
                plot.update(title='loss', x_val=epoch, y_val=[[train_loss, val_loss]])

# 将训练过程的数据保存到文件中
stat.dumpTrainingResult()

plotLine(stat.getHistAcc(), ('train acc', 'val acc'),
         title=model_name+' accuracy',
         gap=ValCycle,
         color_list=('blue', 'red'),
         style_list=('-','-'),
         save_path=train_path_manager.Doc()+'acc.png')

plotLine(stat.getHistLoss(), ('train loss', 'val loss'),
         title=model_name+' loss',
         gap=ValCycle,
         color_list=('blue', 'red'),
         style_list=('-','-'),
         save_path=train_path_manager.Doc()+'loss.png')

########################################################################################
# model.train()
# model.zero_grad()
#
# loss_val = t.zeros((1,)).cuda()
# acc_val = 0.
#
# for task_i in range(taskBatchSize):
#     if TrainingVerbose:
#         print('forming episode...')
#     model_input, labels = train_task.episode()#support, query, sup_len, que_len, labels = train_task.episode()
#     # support, query, labels = train_task.episode()
#
#     if TrainingVerbose:
#         print('forwarding...')
#
#     # print(model_input[0].size(), model_input[1].size())
#
#     predicts = model(*model_input)
#
#     loss_val += loss(predicts, labels)
#
#     predicts = predicts.cpu()
#     acc_val += train_task.accuracy(predicts)
#
# if TrainingVerbose:
#     print('backward...')
# loss_val.backward()
#
# if TrainingVerbose:
#     print('recording grad...')
#
# if TrainingVerbose:
#     print('optimizing...')
#
# optimizer.step()
# scheduler.step()
#
# loss_val_item = loss_val.detach().item()
########################################################################################


# "ATAML在加权以后，将sequence维度相加约减，而不是dim维度约减",
# "3e-2 inner-loop 学习率",
# "修复了ATAML的bug，该bug导致多次adapt只有最后一次adapt有效",
# "每个 inner loop 进行1次adapt",
# "qk减小到5"
