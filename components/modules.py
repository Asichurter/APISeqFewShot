import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math

from components.sequence.LSTM import BiLstmEncoder
from utils.matrix import batchDot
from utils.training import getMaskFromLens


#########################################
# 自注意力模块。输入一个批次的序列输入，得到序列
# 的自注意力对齐结构结果，返回的还是一个等长的序列。
# 使用的是两个全连接层，使用tanh激活。
#########################################
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, pack=True):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################
        # self.AttInter = nn.Linear(input_size, hidden_size, bias=False)
        # self.AttExt = nn.Linear(hidden_size, 1, bias=False)
        ################################################################

        ################################################################
        self.Q = nn.Linear(input_size, input_size, bias=False)
        self.K = nn.Linear(input_size, input_size, bias=False)
        self.V = nn.Linear(input_size, input_size, bias=False)
        ################################################################

        # self.Pack = pack

    def forward(self, x):
        packed = isinstance(x, t.nn.utils.rnn.PackedSequence)
        if packed:
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

            mask = getMaskFromLens(lens)
            # max_idx = lens[0]
            # batch_size = len(lens)
            # idx_matrix = t.arange(0, max_idx, 1).repeat((batch_size, 1))
            # len_mask = lens.unsqueeze(1)
            # mask = idx_matrix.ge(len_mask).cuda()

        assert len(x.size()) == 3, '自注意力输入必须满足(batch, seq, feature)形式！'
        feature_dim = x.size(2)

        # weight shape: [batch, seq, 1]
        ################################################################
        # att_weight = self.AttExt(t.tanh(self.AttInter(x))).squeeze()    # TODO: 根据长度信息来对长度以外的权重进行mask
        ################################################################

        ################################################################
        # shape: [batch, seq, dim]
        att_weight = batchDot(self.Q(x), self.K(x), transpose=True) / math.sqrt(x.size(2))
        ################################################################

        if packed:
            att_weight.masked_fill_(mask, float('-inf'))

        # 自注意力概率分布系数，对序列隐藏态h进行加权求和
        att_weight = t.softmax(att_weight, dim=2)
        # att_weight = t.softmax(att_weight, dim=1).unsqueeze(-1).repeat((1,1,feature_dim))

        return batchDot(att_weight, self.V(x))
        # return (att_weight * x).sum(dim=1)


##########################################################
# 带有残差连接的
##########################################################
class ResInception(nn.Module):

    def __init__(self, in_channel,
                 out_channel,
                 depth=3,
                 reduced_channels=None):
        '''
            注意：out_channel是指每一条路径的输出通道数量，实际的
            总通道输出数量是4×out_channel，因为有4条路径：
            1. 残差连接\n
            2. 1×1卷积\n
            3. 3×3卷积\n
            4. 5×5卷积
        '''

        # TODO：1×1卷积的降维实现
        super(ResInception, self).__init__()

        self.Shortcut = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,
                                  out_channels=out_channel,
                                  kernel_size=(1,1,1),
                                  stride=1,
                                  padding=0,
                                  bias=False),
            nn.BatchNorm3d(out_channel)
        )

        self.Conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=(depth,1,1),
                      stride=1,
                      padding=(depth//2,0,0),
                      bias=False),
            nn.BatchNorm3d(out_channel)
        )

        self.Conv3x3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=(depth, 1, 1),
                      stride=1,
                      padding=(depth // 2, 0, 0),
                      bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Conv3d(in_channels=out_channel,
                      out_channels=out_channel,
                      kernel_size=(depth,3,3),
                      stride=1,
                      padding=(depth//2,1,1),
                      bias=False),
            nn.BatchNorm3d(out_channel)
        )

        self.Conv5x5 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=(depth, 1, 1),
                      stride=1,
                      padding=(depth // 2, 0, 0),
                      bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Conv3d(in_channels=out_channel,
                      out_channels=out_channel,
                      kernel_size=(depth,5,5),
                      stride=1,
                      padding=(depth//2,2,2),
                      bias=False),
            nn.BatchNorm3d(out_channel)
        )

        self.Pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,3,3),
                         stride=1,
                         padding=(0,1,1)),
            nn.Conv3d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=(depth,1,1),
                      stride=1,
                      padding=(depth//2,0,0),
                      bias=False),
            nn.BatchNorm3d(out_channel)
        )


    def forward(self, x):
        # x shape: [batch, in_channel=1, seq, height, width]
        x_1 = self.Conv1x1(x)
        x_3 = self.Conv3x3(x)
        x_5 = self.Conv5x5(x)
        x = self.Shortcut(x)

        # 按channel维度连接所有特征
        return t.cat((x, x_1, x_3, x_5), dim=1)

def CNNBlock2D(in_feature, out_feature, stride=1, kernel=3, padding=1,
             relu='relu', pool='max', flatten=None):
    layers = [nn.Conv2d(in_feature, out_feature,
                  kernel_size=kernel,
                  padding=padding,
                  stride=stride,
                  bias=False),
            nn.BatchNorm2d(out_feature)]

    if relu == 'relu' or relu == True:
        layers.append(nn.ReLU(inplace=True))
    elif relu == 'leaky':
        layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

    if pool == 'max':
        layers.append(nn.MaxPool2d(2))
    elif pool == 'ada':
        layers.append(nn.AdaptiveMaxPool2d(1))

    if flatten:
        layers.append(nn.Flatten(start_dim=flatten))

    return nn.Sequential(*layers)


###############################################################
# 参考HATT-ProtoNet中的Encoding实现，使用CNN在序列维度上进行卷积来
# 解码从RNN中提取到的特征
###############################################################
class CNNEncoder2D(nn.Module):
    def __init__(self,
                 dims,
                 kernel_sizes=[3],
                 paddings=[1],
                 relus=[True],
                 pools=['ada']):
        super(CNNEncoder2D, self).__init__()

        layers = [CNNBlock2D(dims[i], dims[i+1],
                             kernel=kernel_sizes[i],
                             padding=paddings[i],
                             relu=relus[i],
                             pool=pools[i],
                             flatten= i==len(dims)-2)
                  for i in range(len(dims)-1)]

        self.Encoder = nn.Sequential(*layers)

    def forward(self, x, lens=None, transposed=True):
        # input shape: [batch, seq, dim]
        x = x.unsqueeze(1)
        if transposed:
            x = x.transpose(1,2).contiguous()
        x = self.Encoder(x)
        return x.squeeze()


class CnnNGramEncoder(nn.Module):
    def __init__(self,
                 dims,
                 kernel_sizes=[3],
                 paddings=[1],
                 relus=[True],
                 pools=['ada']):
        super(CnnNGramEncoder, self).__init__()

        layers = [CNNBlock2D(dims[i], dims[i+1],
                             kernel=kernel_sizes[i],
                             padding=paddings[i],
                             relu=relus[i],
                             pool=None)
                  for i in range(len(dims)-1)]

        self.Encoder = nn.Sequential(*layers)

    def forward(self, x, lens=None):
        # input shape: [batch, 1, seq, dim]
        seq_len = x.size(1)
        x = x.unsqueeze(1)
        x = self.Encoder(x)

        x = F.adaptive_max_pool2d(x, (seq_len,1))

        # shape: [batch, channel, seq, dim=1] => [batch, seq, channel(dim=1)]
        return x.squeeze().transpose(1,2).contiguous().flatten(start_dim=2)



class CNNEncoder(nn.Module):

    def __init__(self,
                 channels,          # 各个层（包括输入层）的通道数
                 strides=None,      # 各个层的步长
                 flatten=False,
                 relus=None,
                 pools=None):    # 是否在输出前将序列长度展开到特征层中
        super(CNNEncoder, self).__init__()

        self.Flatten = flatten
        assert channels[0]==1, '2D的CNN输入通道必须为1'

        if strides is None:
            strides = [1]*(len(channels)-1)
        if relus is None:
            relus = [True]*(len(channels)-1)
        if pools is None:
            pools = [True]*(len(channels)-1)

        layers = nn.ModuleList([CNNBlock(in_feature=channels[i],
                           out_feature=channels[i+1],
                           stride=strides[i],
                           relu=relus[i],
                           pool=pools[i])
                  for i in range(len(channels)-1)])
        self.Encoder = layers

    def forward(self, x):
        # 假定输入是矩阵序列，本模块将会把序列长度合并到batch中
        # input shape: [batch, seq, height, width]
        assert len(x.size())==4, '%s'%str(x.size())
        batch, seq_len, height, width = x.size()

        x = x.view(batch*seq_len, 1, height, width)
        for layer in self.Encoder:
            x = layer(x)

        if self.Flatten:
            x = x.view(batch, -1)
        else:
            x = x.view(batch, seq_len, -1)

        return x

#####################################################
# Neural Tensor Layer, 用于建模向量关系，实质是一个双线性层
#####################################################
class NTN(nn.Module):
    def __init__(self, c, e, k):
        super(NTN, self).__init__()
        self.Bilinear = nn.Bilinear(c, e, k, bias=False)
        self.Scoring = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(k ,1, bias=True),
        )

    def forward(self, c, e, n):
        v = self.Bilinear(c, e)
        s = self.Scoring(v)
        s = t.sigmoid(s)#t.log_softmax(s.view(-1,n), dim=1)
        return s


class ResDenseLayer(nn.Module):

    def __init__(self, dim):
        super(ResDenseLayer, self).__init__()
        self.Weight = nn.Linear(dim, dim)

    def forward(self, x):
        return t.relu(x+self.Weight(x))


#####################################################
# 基于序列时间步仿射变换的任务嵌入结构，用于向模型中提供任务信息。
# 提供任务原型后生成仿射权重和仿射偏置，作用在序列的每一个step上。
# 主要由残差全连接层构成，还有一个范数惩罚的后乘子
#####################################################
class TenAffine1D(nn.Module):

    def __init__(self, task_dim, feature_dim, layer_num=3):
        super(TenAffine1D, self).__init__()
        
        assert layer_num > 1

        # the weight TEN generate the vector of length equal to
        # sequence length, to weight each step
        weight_ten = [nn.Linear(task_dim, feature_dim),
                      nn.ReLU(inplace=True)]
        for i in range(layer_num-1):
            weight_ten.append(ResDenseLayer(feature_dim))
        self.WeightTEN = nn.Sequential(*weight_ten)


        bias_ten = [nn.Linear(task_dim, feature_dim),
                    nn.ReLU(inplace=True)]
        for i in range(layer_num-1):
            bias_ten.append(ResDenseLayer(feature_dim))
        self.BiasTEN = nn.Sequential(*bias_ten)

        self.WeightMuplier = nn.Parameter(t.empty((feature_dim,)))
        nn.init.normal_(self.WeightMuplier.data, mean=0, std=0.01)

        self.BiasMuplier = nn.Parameter(t.empty((feature_dim,)))
        nn.init.normal_(self.BiasMuplier.data, mean=0, std=0.01)

    def forward(self, x, task_proto=None):

        # not feed task proto, identity mapping
        if task_proto is None:
            return x

        else:
            # shape: [step_len,]
            weight = self.WeightTEN(task_proto)
            bias = self.BiasTEN(task_proto)

            # ----------------------------step-wise affine -------------------------------------
            # x_batch_size, x_feature_dim = x.size(0), x.size(2)
            #
            # # post-multiplier is designed to operate step-wise multiply
            # weight = (weight * self.WeightMuplier) + t.ones_like(weight, device='cuda:0')
            # bias = bias * self.BiasMuplier
            #
            # # weight and bias are applied on each step channel
            # # identically for each sample
            # weight = weight.unsqueeze(1).repeat(1,x_feature_dim).repeat(x_batch_size,1,1)
            # bias = bias.unsqueeze(1).repeat(1,x_feature_dim).repeat(x_batch_size,1,1)
            #--------------------------------------------------------------------------

            # ----------------------------feature-wise affine -------------------------------------
            weight = weight.expand_as(x)
            bias = bias.expand_as(x)
            #--------------------------------------------------------------------------


            return weight*x + bias

    def penalizedNorm(self):
        return self.WeightMuplier.norm(), self.BiasMuplier.norm()





if __name__ == '__main__':
    model = BiLstmEncoder(input_size=64)













