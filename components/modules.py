import torch as t
import torch.nn as nn

#########################################
# 自注意力模块。输入一个批次的序列输入，得到序列
# 的自注意力对齐结构，返回序列的解码结果。
# 使用的是两个全连接层，使用tanh激活。
#########################################
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.AttInter = nn.Linear(input_size, hidden_size)
        self.AttExt = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        assert len(x.size()) == 3, '自注意力输入必须满足(batch, seq, feature)形式！'
        feature_dim = x.size(2)

        # weight shape: [batch, seq, 1]
        att_weight = self.AttExt(t.tanh(self.AttInter(x))).squeeze()

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

        self.Encoder = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=layer_num,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=True)

        if self_attention:
            self.Attention = SelfAttention(2*hidden_size, self_att_dim)
        else:
            self.Attention = None

    def forward(self, x):
        # x shape: [batch, seq, feature]
        # out shape: [batch, seq, 2*hidden]
        out, (h, c) = self.Encoder(x)

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








