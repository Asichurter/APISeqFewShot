import torch as t
import torch.nn as nn
import math

from utils.matrix import batchDot
from utils.training import getMaskFromLens, unpackAndMean

#########################################
# 自注意力模块。输入一个批次的序列输入，得到序列
# 的自注意力对齐结构，返回序列的解码结果。
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


class AttnReduction(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(AttnReduction, self).__init__()

        self.IntAtt = nn.Linear(input_dim, hidden_dim, bias=False)
        self.ExtAtt = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, lens=None):
        if isinstance(x, t.nn.utils.rnn.PackedSequence):
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        feature_dim = x.size(2)

        # weight shape: [batch, seq, 1]
        att_weight = self.ExtAtt(t.tanh(self.IntAtt(x))).squeeze()    # TODO: 根据长度信息来对长度以外的权重进行mask

        if lens is not None:
            if not isinstance(lens, t.Tensor):
                lens = t.Tensor(lens)
            # max_idx = max(lens)#lens[0]
            # batch_size = len(lens)
            # idx_matrix = t.arange(0, max_idx, 1).repeat((batch_size, 1))
            # len_mask = lens.unsqueeze(1)
            # mask = idx_matrix.ge(len_mask).cuda()
            mask = getMaskFromLens(lens)
            att_weight.masked_fill_(mask, float('-inf'))

        att_weight = t.softmax(att_weight, dim=1).unsqueeze(-1).repeat((1,1,feature_dim))
        return (att_weight * x).sum(dim=1)




#########################################
# 双向LTSM并支持自注意力的序列解码器。返回一个
# 由双向序列隐藏态自注意力对齐得到的编码向量。
#########################################
class BiLstmEncoder(nn.Module):
    def __init__(self, input_size,
                 hidden_size=128,
                 layer_num=1,
                 dropout=0.1,
                 self_attention=True,
                 self_att_dim=64,
                 useBN=False):

        super(BiLstmEncoder, self).__init__()

        self.SelfAtt = self_attention
        self.UseBN = useBN

        self.Encoder = nn.GRU(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=layer_num,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=True)

        if useBN:
            # 第一个批标准化建立在时间序列组成的2D矩阵上，扩增了一个维度为1的通道
            # 同时因为序列长度不一定，不能直接在序列长度上进行1D标准化
            self.BN1 = nn.BatchNorm2d(1)
            # 第二个批标准化建立在自注意力之后的1D向量上
            self.BN2 = nn.BatchNorm1d(2*hidden_size)

        if self_attention:
            self.Attention = AttnReduction(2*hidden_size, self_att_dim)
        else:
            self.Attention = None

    def forward(self, x, lens=None):
        # x shape: [batch, seq, feature]
        # out shape: [batch, seq, 2*hidden]
        out, h = self.Encoder(x)
        if self.UseBN:
            out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # 增加一个通道维度以便进行2D标准化
            out = out.unsqueeze(1)
            out = self.BN1(out).squeeze()
            out = nn.utils.rnn.pack_padded_sequence(out, lens, batch_first=True)
        # out, (h, c) = self.Encoder(x)

        # return shape: [batch, feature]
        if self.Attention is not None:
            out = unpackAndMean(out)
            # out = self.Attention(out)             # TODO: 使用简单的平均值代替注意力
            if self.UseBN:
                out = self.BN2(out)
            return out
        else:
            # 没有自注意力时，返回最后一个隐藏态
            num_directions = 2 if self.Encoder.bidirectional else 1
            batch_size = h.size(1)
            h = h.view(self.Encoder.num_layers,
                       num_directions,
                       batch_size,
                       self.Encoder.hidden_size)

            # 取最后一个隐藏态的最后一层的所有方向的拼接向量
            return h[-1].transpose(0,1).contiguous().view(batch_size, self.Encoder.hidden_size*num_directions)


class TransformerEncoder(nn.Module):
    def __init__(self, layer_num, embedding_size, feature_size, att_hid=128, head_size=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.ForwardTrans = nn.Linear(embedding_size, feature_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size,
                                                   nhead=head_size,
                                                   dropout=dropout,
                                                   dim_feedforward=256)

        self.Encoder = nn.TransformerEncoder(encoder_layer, layer_num)

        self.PositionEncoding = PositionalEncoding(embedding_size, dropout=dropout)

        self.Attention = AttnReduction(input_dim=feature_size, hidden_dim=att_hid)

    def forward(self, x, lens):
        x = self.ForwardTrans(x)

        # shape: [seq, batch, dim]
        # 由于transformer要序列优先，因此对于batch优先的输入先进行转置
        x = x.transpose(0,1).contiguous()
        max_len = int(lens[0])
        mask = t.Tensor([[0 if i < j else 1 for i in range(int(max_len))] for j in lens]).bool().cuda()
        x = self.PositionEncoding(x)
        x = self.Encoder(src=x,
                         src_key_padding_mask=mask)          # TODO:根据lens长度信息构建mask输入到transformer中
        x = x.transpose(0,1).contiguous()

        x = self.Attention(x, lens=lens)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=4000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 由于PositionEncoding位于Transformer中，因此seq先于batch
        # shape: [seq, batch, dim]
        max_len = x.size(0)
        d_model = x.size(2)
        bacth_size = x.size(1)

        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.repeat((bacth_size,1,1)).transpose(0, 1).cuda()

        x = x + pe

        return self.dropout(x)

class BiLstmCellLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 bidirectional=True):

        super(BiLstmCellLayer, self).__init__()

        self.Bidirectional = bidirectional

        self.ForwardCell = nn.LSTMCell(input_size=input_size,
                                  hidden_size=hidden_size)

        if bidirectional:
            self.BackwardCell = nn.LSTMCell(input_size=input_size,
                                  hidden_size=hidden_size)
        else:
            self.BackwardCell = None

    def forward(self, x):
        assert not self.Bidirectional or type(x)==tuple, \
            '双向LSTM单元的输入必须是正向和反向两个输入'

        if self.Bidirectional:
            forward_x = x[0]
            backward_x = x[1]
        else:
            forward_x = x

        # input shape: [batch, seq, dim]
        num_directions = 2 if self.Bidirectional else 1
        batch_size = forward_x.size(0)
        seq_len = forward_x.size(1)
        hidden_dim = self.ForwardCell.hidden_size

        forward_hidden_states = t.empty((batch_size, seq_len, hidden_dim)).cuda()
        if self.Bidirectional:
            backward_hidden_states = t.empty((batch_size, seq_len, hidden_dim)).cuda()

        # 定义初始状态为0向量
        f_h_x, f_c_x = t.zeros((batch_size, hidden_dim)).cuda(), t.zeros((batch_size, hidden_dim)).cuda()
        if num_directions > 1:
            b_h_x, b_c_x = t.zeros((batch_size, hidden_dim)).cuda(), t.zeros((batch_size, hidden_dim)).cuda()

        for i in range(seq_len):
            f_h_x, f_c_x = self.ForwardCell(forward_x[:,i,:], (f_h_x, f_c_x))
            forward_hidden_states[:,i,:] = f_h_x

            if num_directions > 1:
                b_h_x, b_c_x = self.BackwardCell(backward_x[:,seq_len-1-i,:], (b_h_x, b_c_x))
                # 反向的序列需要将隐藏层放置在首位使得与正向隐藏态对齐
                backward_hidden_states[:,seq_len-1-i,:] = b_h_x

        if self.Bidirectional:
            return forward_hidden_states, backward_hidden_states
        else:
            return forward_hidden_states

class BiLstmCellEncoder(nn.Module):

    def __init__(self, input_size,
                 hidden_size=128,
                 num_layers=1,
                 bidirectional=True,
                 self_att_dim=64):
        super(BiLstmCellEncoder, self).__init__()

        self.Bidirectional = bidirectional

        layers = [BiLstmCellLayer(input_size=input_size if i==0 else hidden_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional)
                  for i in range(num_layers)]

        self.LstmCells = nn.Sequential(*layers)

        self.SelfAttention = SelfAttention(input_size=hidden_size*2 if bidirectional else hidden_size,
                                           hidden_size=self_att_dim,
                                           pack=False)


    def forward(self, x):
        bacth_size = x.size(0)
        seq_len = x.size(1)
        hidden_size = self.SelfAttention.input_size

        if self.Bidirectional:
            f_h, b_h = self.LstmCells((x,x))
            x = t.cat((f_h, b_h), dim=2)
            x = x.view(bacth_size, seq_len, hidden_size)
        else:
            x = self.LstmCells(x)

        x = self.SelfAttention(x)
        return x

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

def CNNBlock(in_feature, out_feature, stride=1, kernel=3, padding=1,
             relu=True, pool=True):
    layers = [nn.Conv2d(in_feature, out_feature,
                  kernel_size=kernel,
                  padding=padding,
                  stride=stride,
                  bias=False),
            nn.BatchNorm2d(out_feature)]

    if relu:
        layers.append(nn.ReLU(inplace=True))
    if pool:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)



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


if __name__ == '__main__':
    model = ResInception(1,1)
    a = 0













