import torch as t
from torch import nn as nn
from torch.nn import _VF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from components.reduction.selfatt import AttnReduction


#########################################
# 双向LTSM并支持自注意力的序列解码器。返回一个
# 由双向序列隐藏态自注意力对齐得到的编码向量。
#########################################
class BiLstmEncoder(nn.Module):
    def __init__(self, input_size=None,
                 hidden_size=128,
                 num_layers=1,
                 dropout=0.1,
                 self_att_dim=None,
                 useBN=False,
                 sequential=False,
                 bidirectional=True,
                 **kwargs):

        super(BiLstmEncoder, self).__init__()

        self.SelfAtt = self_att_dim is not None
        self.UseBN = useBN
        self.Sequential = sequential

        self.Encoder = nn.LSTM(input_size=input_size,  # GRU
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)

        if useBN:
            # 第一个批标准化建立在时间序列组成的2D矩阵上，扩增了一个维度为1的通道
            # 同时因为序列长度不一定，不能直接在序列长度上进行1D标准化
            self.BN1 = nn.BatchNorm2d(1)
            # 第二个批标准化建立在自注意力之后的1D向量上
            self.BN2 = nn.BatchNorm1d(2*hidden_size)

        if self.SelfAtt:
            self.Attention = AttnReduction((1 + bidirectional) * hidden_size, self_att_dim)
            # self.Attention = AttnReduction(2*hidden_size, self_att_dim)
        else:
            self.Attention = None

    def forward(self, x, lens):
        if not isinstance(x, t.nn.utils.rnn.PackedSequence) and lens is not None:
            x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        # x shape: [batch, seq, feature]
        # out shape: [batch, seq, 2*hidden]
        out, h = self.Encoder(x)
        if self.UseBN:
            out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # 增加一个通道维度以便进行2D标准化
            out = out.unsqueeze(1)
            out = self.BN1(out).squeeze()
            out = nn.utils.rnn.pack_padded_sequence(out, lens, batch_first=True, enforce_sorted=False)
        # out, (h, c) = self.Encoder(x)

        # return shape: [batch, feature]
        if self.Attention is not None:
            # out = unpackAndMean(out)
            out = self.Attention(out)
            if self.UseBN:
                out = self.BN2(out)

        else:
            # 由于使用了CNN进行解码，因此还是可以返回整个序列
            out, lens = pad_packed_sequence(out, batch_first=True)

        if self.Sequential:
            return out, lens
        else:
            return out

    @staticmethod
    def permute_hidden(hx, permutation):
        # type: (Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        if permutation is None:
            return hx
        return BiLstmEncoder.apply_permutation(hx[0], permutation), \
               BiLstmEncoder.apply_permutation(hx[1], permutation)

    @staticmethod
    def apply_permutation(tensor, permutation, dim=1):
        # type: (Tensor, Tensor, int) -> Tensor
        return tensor.index_select(dim, permutation)

    #################################################
    # 使用给定的参数进行forward
    #################################################
    def static_forward(self, x, lens, params):           # PyTorch1.4目前不支持在rnn上多次backward
        packed = isinstance(x, t.nn.utils.rnn.PackedSequence)
        if not packed and lens is not None:
            x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        x, batch_sizes, sorted_indices, unsorted_indices = x
        max_batch_size = int(batch_sizes[0])

        num_directions = 2
        zeros = t.zeros(self.Encoder.num_layers * num_directions,
                            max_batch_size, self.Encoder.hidden_size,
                            dtype=x.dtype, device=x.device)
        hx = (zeros, zeros)

        weights = [params['Encoder.Encoder.weight_ih_l0'],
                   params['Encoder.Encoder.weight_hh_l0'],
                   params['Encoder.Encoder.bias_ih_l0'],
                   params['Encoder.Encoder.bias_hh_l0'],
                   params['Encoder.Encoder.weight_ih_l0_reverse'],
                   params['Encoder.Encoder.weight_hh_l0_reverse'],
                   params['Encoder.Encoder.bias_ih_l0_reverse'],
                   params['Encoder.Encoder.bias_hh_l0_reverse']
                   ]

        result = _VF.lstm(x, batch_sizes, hx,
                          weights,
                          True,     # 是否bias
                          self.Encoder.num_layers,
                          self.Encoder.dropout,
                          self.Encoder.training,
                          self.Encoder.bidirectional)

        out, h = result[0], result[1:]

        out = PackedSequence(out, batch_sizes, sorted_indices, unsorted_indices)
        h = BiLstmEncoder.permute_hidden(h, unsorted_indices)

        if self.Attention is not None:
            out = AttnReduction.static_forward(out, params)
            return out
        else:
            # TODO: 由于使用了CNN进行解码，因此还是可以返回整个序列
            out, lens = pad_packed_sequence(out, batch_first=True)
            return out




            # 没有自注意力时，返回最后一个隐藏态
            # num_directions = 2 if self.Encoder.bidirectional else 1
            # batch_size = h.size(1)
            # h = h.view(self.Encoder.num_layers,
            #            num_directions,
            #            batch_size,
            #            self.Encoder.hidden_size)
            # 取最后一个隐藏态的最后一层的所有方向的拼接向量
            # return h[-1].transpose(0,1).contiguous().view(batch_size, self.Encoder.hidden_size*num_directions)


if __name__ == '__main__':
    m = BiLstmEncoder(1)