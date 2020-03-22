import torch as t
import torch.nn as nn

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
        self.AttInter = nn.Linear(input_size, hidden_size, bias=False)
        self.AttExt = nn.Linear(hidden_size, 1, bias=False)
        self.Pack = pack

    def forward(self, x):
        if isinstance(x, t.nn.utils.rnn.PackedSequence):
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

            # max_idx = lens[0]
            # batch_size = len(lens)
            # idx_matrix = t.arange(0, max_idx, 1).repeat((batch_size, 1))
            # len_mask = lens.unsqueeze(1)
            # mask = idx_matrix.ge(len_mask).cuda()

        assert len(x.size()) == 3, '自注意力输入必须满足(batch, seq, feature)形式！'
        feature_dim = x.size(2)

        # weight shape: [batch, seq, 1]
        att_weight = self.AttExt(t.tanh(self.AttInter(x))).squeeze()    # TODO: 根据长度信息来对长度以外的权重进行mask

        # att_weight.masked_fill_(mask, -1*float('inf'))

        # 自注意力概率分布系数，对序列隐藏态h进行加权求和
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
                 self_att_dim=64):

        super(BiLstmEncoder, self).__init__()

        self.SelfAtt = self_attention

        self.Encoder = nn.GRU(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=layer_num,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=True)

        if self_attention:
            self.Attention = SelfAttention(2*hidden_size, self_att_dim, pack=True)
        else:
            self.Attention = None

    def forward(self, x):
        # x shape: [batch, seq, feature]
        # out shape: [batch, seq, 2*hidden]
        out, h = self.Encoder(x)
        # out, (h, c) = self.Encoder(x)

        # return shape: [batch, feature]
        if self.Attention is not None:
            return self.Attention(out)
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

class CNNBlock(nn.Module):
    def __init__(self, in_chanel, out_channel, kernel_size, stride, padding, pool_size=None, bn=True):
        self.CNN = nn.Conv2d(in_channels=in_chanel,
                             out_channels=out_channel,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             bias=False)
        self.BN = nn.BatchNorm2d(out_channel) if bn else None
        self.Pool = nn.AdaptiveMaxPool2d(())

class AttentionalCNN(nn.Module):

    def __init__(self, window_size=3, channels=[1,32,1]):
        super(AttentionalCNN, self).__init__()

        layers = [nn.Conv2d(in_channels=channels[i],
                            out_channels=channels[i+1],
                            kernel_size=kernel_size,
                            stride=1,
                            padding= kernel_size//2,
                            bias=False) for i in range(len(channels)-1)]








