import torch as t
from torch import nn as nn
from torch.nn import functional as F

from utils.training import getMaskFromLens


class AttnReduction(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, max_seq_len=200,
                 **kwargs):
        super(AttnReduction, self).__init__()

        self.MaxSeqLen = max_seq_len

        self.IntAtt = nn.Linear(input_dim, input_dim, bias=False)
        self.ExtAtt = nn.Linear(input_dim, 1, bias=False)

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
            mask = getMaskFromLens(lens,self.MaxSeqLen)
            att_weight.masked_fill_(mask, float('-inf'))

        att_weight = t.softmax(att_weight, dim=1).unsqueeze(-1).repeat((1,1,feature_dim))
        return (att_weight * x).sum(dim=1)

    @staticmethod
    def static_forward(x, params, lens=None):               # 此处由于命名限制，假定参数是按照使用顺序feed进来的
        if isinstance(x, t.nn.utils.rnn.PackedSequence):
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        feature_dim = x.size(2)

        att_weight = F.linear(input=t.tanh(F.linear(input=x,
                                                    weight=params[0])),
                              weight=params[1]).squeeze()

        if lens is not None:
            if not isinstance(lens, t.Tensor):
                lens = t.Tensor(lens)
            mask = getMaskFromLens(lens)
            att_weight.masked_fill_(mask, float('-inf'))

        att_weight = t.softmax(att_weight, dim=1).unsqueeze(-1).repeat((1,1,feature_dim))
        return (att_weight * x).sum(dim=1)