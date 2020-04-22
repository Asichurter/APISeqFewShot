import os
import sys
import shutil
from config import appendProjectPath, saveConfigFile

################################################
#----------------------设置系统基本信息------------------
################################################

appendProjectPath(depth=1)

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

from models.ProtoNet import ProtoNet, ImageProtoNet, IncepProtoNet, CNNLstmProtoNet
from models.InductionNet import InductionNet
from models.MetaSGD import MetaSGD


################################################
#----------------------读取参数------------------
################################################

data_folder = cfg.dataset()#'virushare_20_image'

k,n,qk,N = cfg.taskParams()

LRDecayIters,\
LRDecayGamma,\
optimizer_type,\
weight_decay, \
loss_func,\
default_lr,\
lrs, \
taskBatchSize = cfg.trainingParams()

EmbedSize,\
HiddenSize,\
BiLstmLayer,\
SelfAttDim, \
usePretrained,\
wordCnt = cfg.modelParams()

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

# train_task = ProtoEpisodeTask(k, qk, n, N, train_dataset,
#                               cuda=True, expand=expand)
# val_task = ProtoEpisodeTask(k, qk, n, N, val_dataset,
#                               cuda=True, expand=expand)

train_task = AdaptEpisodeTask(k, qk, n, N, train_dataset,
                              cuda=True, expand=expand)
val_task = AdaptEpisodeTask(k, qk, n, N, val_dataset,
                              cuda=True, expand=expand)

stat = TrainStatManager(model_save_path=train_path_manager.Model(),
                        train_report_iter=ValCycle,
                        criteria='loss')

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

if usePretrained:
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
                    lr=default_lr,
                    pretrained_matrix=word_matrix,
                    embed_size=EmbedSize
                    # hidden_size=HiddenSize,
                    # layer_num=BiLstmLayer,
                    # self_att_dim=SelfAttDim
                    # word_cnt=wordCnt,
                    # freeze_embedding=False
                    )
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
    optimizer = t.optim.SGD(parameters)
else:
    raise ValueError

scheduler = t.optim.lr_scheduler.StepLR(optimizer,
                                        step_size=LRDecayIters,
                                        gamma=LRDecayGamma)


################################################
#----------------------训练------------------
################################################


grad = 0.

printState('start training')
stat.startTimer()

# 保存配置文件到doc
saveConfigFile(train_path_manager.Doc())

with t.autograd.set_detect_anomaly(False):
    for epoch in range(TrainingEpoch):
        # print('Epoch', epoch)
        if TrainingVerbose:
            printState('Epoch %d'%epoch)

        model.train()
        model.zero_grad()

        loss_val = t.zeros((1,)).cuda()
        acc_val = 0.

        for task_i in range(taskBatchSize):
            if TrainingVerbose:
                print('forming episode...')
            model_input, labels = train_task.episode()#support, query, sup_len, que_len, labels = train_task.episode()
            # support, query, labels = train_task.episode()

            if TrainingVerbose:
                print('forwarding...')

            # print(model_input[0].size(), model_input[1].size())

            predicts = model(*model_input)

            loss_val += loss(predicts, labels)

            predicts = predicts.cpu()
            acc_val += train_task.accuracy(predicts)

        if TrainingVerbose:
            print('backward...')
        loss_val.backward()

        if TrainingVerbose:
            print('recording grad...')

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

        if TrainingVerbose:
            print('optimizing...')

        optimizer.step()
        scheduler.step()

        loss_val_item = loss_val.detach().item()

        if TrainingVerbose:
            print('recording...')

        # 记录任务batch的平均正确率和损失值
        stat.recordTraining(acc_val / taskBatchSize,
                            loss_val_item/ taskBatchSize)


        ################################################
        # ----------------------验证------------------
        ################################################

        if epoch % ValCycle == 0:
            printState('Test in Epoch %d'%epoch)
            model.eval()
            validate_acc = 0.
            validate_loss = 0.

            for i in range(ValEpisode):
                model_input, labels = val_task.episode()#support, query, sup_len, que_len, labels = val_task.episode()
                # support, query, labels = val_task.episode()

                # print(model_input[0].size(), model_input[1].size())
                predicts = model(*model_input)

                loss_val = loss(predicts, labels)

                predicts = predicts.cpu()
                validate_loss += loss_val.detach().item()
                validate_acc += val_task.accuracy(predicts)

            avg_validate_acc = validate_acc / ValEpisode
            avg_validate_loss = validate_loss / ValEpisode

            stat.recordValidating(avg_validate_acc,
                                  avg_validate_loss,
                                  model)

            train_acc, train_loss, val_acc, val_loss = stat.getRecentRecord()
            if UseVisdom:
                plot.update(title='accuracy', x_val=epoch, y_val=[[train_acc, val_acc]])
                plot.update(title='loss', x_val=epoch, y_val=[[train_loss, val_loss]])


plotLine(stat.getHistAcc(), ('train acc', 'val acc'),
         title=model_name+' accuracy',
         gap=ValCycle,
         color_list=('blue', 'red'),
         style_list=('-','-'),
         ylim=[0,1],
         save_path=train_path_manager.Doc()+'acc.png')

plotLine(stat.getHistLoss(), ('train loss', 'val loss'),
         title=model_name+' loss',
         gap=ValCycle,
         color_list=('blue', 'red'),
         style_list=('-','-'),
         save_path=train_path_manager.Doc()+'loss.png')







