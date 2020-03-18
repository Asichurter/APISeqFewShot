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

    def forward(self, x):
        # x shape: [batch, seq, feature]
        # out shape: [batch, seq, 2*hidden]
        out, _h_c = self.Encoder(x)

        # return shape: [batch, feature]
        return self.Attention(out)







