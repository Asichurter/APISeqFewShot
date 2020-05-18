import torch as t
import torch.nn as nn

from components.sequence.LSTM import BiLstmEncoder


class RelationNet(nn.Module):

    def __init__(self,
                 pretrained_matrix,
                 embed_size,
                 hidden=128,
                 layer_num=1,
                 self_attention=False,
                 self_att_dim=64,
                 word_cnt=None
                 ):

        super(RelationNet, self).__init__()

        # 可训练的嵌入层
        if pretrained_matrix is not None:
            self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        else:
            self.Embedding = nn.Embedding(word_cnt, embedding_dim=embed_size, padding_idx=0)

        self.EmbedNorm = nn.LayerNorm(embed_size)

        self.Encoder = BiLstmEncoder(embed_size,#64
                                     hidden_size=hidden,
                                     layer_num=layer_num,
                                     self_attention=self_attention,
                                     self_att_dim=self_att_dim,
                                     useBN=False)

        self.Relation = nn.Sequential(
        )
