import torch as t
from torch import nn as nn
from torch.nn import _VF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from components.reduction.selfatt import BiliAttnReduction


#########################################
# 双向LTSM并支持自注意力的序列解码器。返回一个
# 由双向序列隐藏态自注意力对齐得到的编码向量。
#########################################
from utils.training import getMaskFromLens


class BiLstmEncoder(nn.Module):
    def __init__(self, input_size=None,
                 hidden_size=128,
                 num_layers=1,
                 dropout=0.1,
                 self_att_dim=None,
                 useBN=False,
                 sequential=False,
                 bidirectional=True,
                 return_last_state=False,
                 max_seq_len=200,
                 **kwargs):

        super(BiLstmEncoder, self).__init__()

        self.SelfAtt = self_att_dim is not None
        self.UseBN = useBN
        self.Sequential = sequential
        self.RetLastSat = return_last_state
        self.MaxSeqLen = max_seq_len

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
            self.Attention = BiliAttnReduction((1 + bidirectional) * hidden_size, self_att_dim)
            # self.Attention = AttnReduction(2*hidden_size, self_att_dim)
        else:
            self.Attention = None

    def forward(self, x, lens):
        if not isinstance(x, t.nn.utils.rnn.PackedSequence) and lens is not None:
            x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        # x shape: [batch, seq, feature]
        # out shape: [batch, seq, 2*hidden]
        out, h = self.Encoder(x)

        # return shape: [batch, feature]
        if self.Attention is not None:
            # out = unpackAndMean(out)
            out = self.Attention(out)

        else:
            # 由于使用了CNN进行解码，因此还是可以返回整个序列
            out, lens = pad_packed_sequence(out, batch_first=True)


            if self.RetLastSat:
                out = out[:,-1,:].squeeze()

            # 如果序列中没有长度等于最大长度的元素,则使用原生pad时会产生尺寸错误
            if out.size(1) != self.MaxSeqLen:
                pad_size = self.MaxSeqLen-out.size(1)
                zero_paddings = t.zeros((out.size(0),pad_size,out.size(2))).cuda()
                out = t.cat((out,zero_paddings),dim=1)

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
            out = BiliAttnReduction.static_forward(out, params)
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


class BiLstmCellEncoder(nn.Module):

    def __init__(self, input_size,
                 hidden_size=128,
                 num_layers=1,
                 bidirectional=True,
                 self_att_dim=64,
                 max_seq_len=200,
                 nonlineary=False,
                 **kwargs):
        super(BiLstmCellEncoder, self).__init__()

        self.Bidirectional = bidirectional
        self.HiddenSize = hidden_size

        layers = [BiLstmCellLayer(input_size=input_size if i==0 else hidden_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  max_seq_len=max_seq_len,
                                  nonlineary=nonlineary)
                  for i in range(num_layers)]

        self.LstmCells = nn.Sequential(*layers)

        # self.SelfAttention = SelfAttention(input_size=hidden_size*2 if bidirectional else hidden_size,
        #                                    hidden_size=self_att_dim,
        #                                    pack=False)


    def forward(self, x, lens=None):
        bacth_size = x.size(0)
        seq_len = x.size(1)
        hidden_size = self.HiddenSize
        directions = self.Bidirectional + 1

        if self.Bidirectional:
            # 由于sequential只能接受一个positional input,因此将输入打包为字典
            f_h, b_h = self.LstmCells({'input':(x,x),
                                       'lens':lens})['input']
            x = t.cat((f_h, b_h), dim=2)
            x = x.view(bacth_size, seq_len, hidden_size*directions)
        else:
            x = self.LstmCells({'input':x,
                                'lens':lens})['input']

        # x = self.SelfAttention(x)
        return x


class BiLstmCellLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 bidirectional=True,
                 max_seq_len=200,
                 nonlineary=None):

        super(BiLstmCellLayer, self).__init__()

        self.Bidirectional = bidirectional
        self.MaxSeqLen = max_seq_len

        self.ForwardCell = nn.LSTMCell(input_size=input_size,
                                  hidden_size=hidden_size)

        if bidirectional:
            self.BackwardCell = nn.LSTMCell(input_size=input_size,
                                  hidden_size=hidden_size)
        else:
            self.BackwardCell = None

        if nonlineary:
            self.Nonlineary = nn.ReLU()
        else:
            self.Nonlineary = nn.Identity()

        self.SequentialVerboseCache = []

    def forward(self, input_dict, sequential_verbose=False):
        assert not self.Bidirectional or type(input_dict['input'])==tuple, \
            '双向LSTM单元的输入必须是正向和反向两个输入'

        if self.Bidirectional:
            forward_x = input_dict['input'][0]
            backward_x = input_dict['input'][1]
        else:
            forward_x = input_dict['input']

        lens = input_dict['lens']

        # input shape: [batch, seq, dim]
        num_directions = self.Bidirectional + 1
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

        sequential_hidden_norm = t.empty((batch_size, seq_len)).cuda()
        sequential_cell_norm = t.empty((batch_size, seq_len)).cuda()

        for i in range(seq_len):
            f_h_x, f_c_x = self.ForwardCell(forward_x[:,i,:], (f_h_x, f_c_x))
            forward_hidden_states[:,i,:] = f_h_x

            if sequential_verbose:
                sequential_hidden_norm[:,i] = t.norm(f_h_x, dim=1).detach()
                sequential_cell_norm[:,i] = t.norm(f_c_x, dim=1).detach()

            if num_directions > 1:
                b_h_x, b_c_x = self.BackwardCell(backward_x[:,seq_len-1-i,:], (b_h_x, b_c_x))
                # 反向的序列需要将隐藏层放置在首位使得与正向隐藏态对齐
                backward_hidden_states[:,seq_len-1-i,:] = b_h_x

        if sequential_verbose:
            self.SequentialVerboseCache = (sequential_hidden_norm, sequential_cell_norm)

        if lens is not None:
            mask = getMaskFromLens(lens, self.MaxSeqLen).unsqueeze(-1).expand_as(forward_hidden_states)
            forward_hidden_states.masked_fill_(mask, 0)

            if self.Bidirectional:
                backward_hidden_states.masked_fill_(mask, 0)

        if self.Bidirectional:
            forward_hidden_states = self.Nonlineary(forward_hidden_states)
            backward_hidden_states = self.Nonlineary(backward_hidden_states)
            return {'input': (forward_hidden_states, backward_hidden_states),
                    'lens': lens}
        else:
            forward_hidden_states = self.Nonlineary(forward_hidden_states)
            return {'input': forward_hidden_states,
                    'lens':lens}