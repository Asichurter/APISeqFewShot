import torch as t
import torch.nn as nn

from components.sequence.transformer import TransformerEncoder


class TransformerSet(nn.Module):

    def __init__(self,
                 trans_input_size,
                 trans_dropout=0.5,
                 trans_head_nums=1,
                 **kwargs):
        super(TransformerSet, self).__init__()

        # self.Transformer = TransformerEncoder(trans_num_layer,
        #                                       trans_input_size,
        #                                       trans_hidden_dim,
        #                                       None,
        #                                       trans_head_size,
        #                                       trans_dropout)
        self.Transformer = nn.MultiheadAttention(embed_dim=trans_input_size,
                                                 num_heads=trans_head_nums,
                                                 dropout=trans_dropout)

        self.fc = nn.Linear(trans_input_size, trans_input_size)
        self.dropout = nn.Dropout(trans_dropout)
        self.layernorm = nn.LayerNorm(trans_input_size)


    def forward(self, x):

        # reshape to [seq, batch, dim]
        x = x.transpose(0,1).contiguous()

        # for set-to-set operation, all sequence item is valid, no padding
        # dummy_lens = [x.size(1)]*x.size(0)

        # input as (query,key,value), namely self-attention
        residual, _weights = self.Transformer(x,x,x)
        # residual = self.Transformer(x, dummy_lens)

        residual = self.dropout(self.fc(residual))

        return self.layernorm(residual + x).transpose(0,1).contiguous()

