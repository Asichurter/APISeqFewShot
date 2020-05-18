import math

import torch as t
from torch import nn as nn

from components.reduction.selfatt import AttnReduction


#################################################
# 利用Transformer进行序列嵌入归纳的Encoder，整合了Position
# Embedding和transformer，得到的序列结果使用自注意力归纳
#################################################
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers,
                 embed_size,
                 hidden_size,
                 self_att_dim=128,
                 head_size=8,
                 dropout=0.1,
                 **kwargs):

        super(TransformerEncoder, self).__init__()

        self.ForwardTrans = nn.Linear(embed_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=head_size,
                                                   dropout=dropout,
                                                   dim_feedforward=256)

        self.Encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.PositionEncoding = PositionalEncoding(embed_size, dropout=dropout)

        self.Attention = AttnReduction(input_dim=hidden_size, hidden_dim=self_att_dim) if self_att_dim is not None else None

    def forward(self, x, lens):
        x = self.ForwardTrans(x)

        # shape: [seq, batch, dim]
        # 由于transformer要序列优先，因此对于batch优先的输入先进行转置
        x = x.transpose(0,1).contiguous()
        max_len = int(max(lens))
        mask = t.Tensor([[0 if i < j else 1 for i in range(int(max_len))] for j in lens]).bool().cuda()
        x = self.PositionEncoding(x)
        # print('\n\nmax_len:', max_len)
        # print('lens:', lens)
        # print('x size:', x.size())
        # print('mask.size:', mask.size())
        x = self.Encoder(src=x,
                         src_key_padding_mask=mask)          # TODO:根据lens长度信息构建mask输入到transformer中
        x = x.transpose(0,1).contiguous()

        if self.Attention is not None:
            x = self.Attention(x, lens)
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