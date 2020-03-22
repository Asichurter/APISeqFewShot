import torch as t
import numpy as np
from pprint import pprint

import sys
sys.path.append('D:/Projects/APISeqFewShot/')       # 添加当前项目路径到包搜索路径中
sys.setrecursionlimit(5000)                         # 增加栈空间防止意外退出

from components.task import ProtoEpisodeTask, ImageProtoEpisodeTask
from utils.manager import PathManager, TrainStatManager
from utils.plot import VisdomPlot
from components.datasets import SeqFileDataset, ImageFileDataset
from models.ProtoNet import ProtoNet, ImageProtoNet
from utils.init import LstmInit
from utils.display import printState
from utils.stat import statParamNumber

data_folder = 'virushare_20'#'virushare_20_image'

Ns = {'cluster':20, 'test':20, 'virushare_20':20, 'drebin_10':10, 'miniImageNet':600, 'virushare_20_image':20}

# 每个类多少个样本，即k-shot
k = 5
# 训练时多少个类参与，即n-way
n = 5
# 测试时每个类多少个样本
qk = 15
# 一个类总共多少个样本
N = Ns[data_folder]

weight_decay = 1e-4

loss_func = 'nll'
expand = False if loss_func=='nll' else True

loss = t.nn.NLLLoss().cuda() \
    if loss_func=='nll' else \
    t.nn.MSELoss().cuda()

optimizer_type = 'adam'

# 学习率
default_lr = 1e-2
lrs = {'Embedding.weight': 1e-3}

EmbedSize = 32
HiddenSize = 64
BiLstmLayer = 2
SelfAttDim = 64
ValCycle = 100
ValEpisode = 50
LRDecayIters = 15000
LRDecayGamma = 0.1
RecordGradient = False
GradientUpdateCycle = 1000
TrainingVerbose = False
UseVisdom = True

TrainingEpoch = 50000

version = 1
model_name = 'ProtoNet_v%d.0' % version

printState('init managers...')
train_path_manager = PathManager(dataset=data_folder, d_type='train', model_name=model_name)
val_path_manager = PathManager(dataset=data_folder, d_type='validate', model_name=model_name)

train_dataset = SeqFileDataset(train_path_manager.FileData(),
                               train_path_manager.FileSeqLen(),
                               N)
val_dataset = SeqFileDataset(val_path_manager.FileData(),
                               val_path_manager.FileSeqLen(),
                               N)
# train_dataset = ImageFileDataset(train_path_manager.FileData(), N, rd_crop_size=224)
# val_dataset = ImageFileDataset(val_path_manager.FileData(), N, rd_crop_size=224)

train_task = ProtoEpisodeTask(k ,qk, n, N,
                        dataset=train_dataset,
                        cuda=True,
                        label_expand=expand)
val_task = ProtoEpisodeTask(k ,qk, n, N,
                        dataset=val_dataset,
                        cuda=True,
                        label_expand=expand)
# train_task = ImageProtoEpisodeTask(k ,qk, n, N,
#                         dataset=train_dataset,
#                         cuda=True,
#                         label_expand=expand)
# val_task = ImageProtoEpisodeTask(k ,qk, n, N,
#                         dataset=val_dataset,
#                         cuda=True,
#                         label_expand=expand)

stat = TrainStatManager(model_save_path=train_path_manager.Model(),
                        train_report_iter=ValCycle,
                        criteria='loss')

types=['line', 'line']
titles=['accuracy', 'loss']
xlabels=['iterations', 'iterations']
ylabels=['accuracy', 'loss']
legends=[['train acc', 'val acc'],
         ['train loss', 'val loss']]

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

word_matrix = t.Tensor(np.load(train_path_manager.WordEmbedMatrix(), allow_pickle=True))

printState('init model...')
model = ProtoNet(pretrained_matrix=word_matrix,
                 embed_size=EmbedSize,
                 hidden=HiddenSize,
                 layer_num=BiLstmLayer,
                 self_attention=SelfAttDim is not None,
                 self_att_dim=SelfAttDim)
# model = ImageProtoNet(in_channels=1)

model = model.cuda()

statParamNumber(model)

# 模型初始化
printState('init parameters...')
model.apply(LstmInit)

parameters = []
for name, par in model.named_parameters():
    # print(name)
    if name in lrs:
        parameters += [{'params': [par], 'lr':lrs[name], 'weight_decay': weight_decay}]
    else:
        parameters += [{'params': [par], 'lr':default_lr, 'weight_decay': weight_decay}]

if optimizer_type == 'adam':
    optimizer = t.optim.Adam(parameters)

scheduler = t.optim.lr_scheduler.StepLR(optimizer,
                                        step_size=LRDecayIters,
                                        gamma=LRDecayGamma)
grad = 0.

printState('start training')
stat.startTimer()

with t.autograd.set_detect_anomaly(False):
    for epoch in range(TrainingEpoch):
        # print('Epoch', epoch)
        if TrainingVerbose:
            printState('Epoch %d'%epoch)

        model.train()
        model.zero_grad()

        if TrainingVerbose:
            print('forming episode...')
        model_input, labels = train_task.episode()#support, query, sup_len, que_len, labels = train_task.episode()
        # support, query, labels = train_task.episode()

        if TrainingVerbose:
            print('forwarding...')

        # print(model_input[0].size(), model_input[1].size())

        predicts = model(*model_input)

        loss_val = loss(predicts, labels)

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
                plot.update('gradient', epoch, [[grad]], update={'flag': True,
                                                         'val': None if epoch%GradientUpdateCycle==0 else 'append'})

        if TrainingVerbose:
            print('optimizing...')
        optimizer.step()
        scheduler.step()

        predicts = predicts.cpu()

        loss_val_item = loss_val.detach().item()

        if TrainingVerbose:
            print('recording...')
        acc_val = train_task.accuracy(predicts)
        stat.recordTraining(acc_val, loss_val_item)

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








