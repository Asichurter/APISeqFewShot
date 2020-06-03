import torch as t
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from components.modules import NTN
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder
from utils.training import dynamicRouting, \
                            extractTaskStructFromInput, \
                            repeatProtoToCompShape, \
                            repeatQueryToCompShape

class InductionNet(nn.Module):

    def __init__(self,
                 pretrained_matrix,
                 embed_size,
                 hidden_size=128,
                 layer_num=1,
                 self_att_dim=64,
                 ntn_hidden=100,
                 routing_iters=3,
                 word_cnt=None,
                 **modelParams):
        super(InductionNet, self).__init__()

        self.Iters = routing_iters

        if pretrained_matrix is not None:
            self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix,
                                                          padding_idx=0)
        else:
            self.Embedding = nn.Embedding(word_cnt, embedding_dim=embed_size, padding_idx=0)

        self.EmbedDrop = nn.Dropout(modelParams['dropout'])

        self.Encoder = BiLstmEncoder(input_size=embed_size,
                                     hidden_size=hidden_size,
                                     layer_num=layer_num,
                                     self_att_dim=self_att_dim,
                                     **modelParams)

        directions = 1 + modelParams['bidirectional']

        self.Decoder = CNNEncoder1D([hidden_size*directions,
                                     hidden_size*directions])

        self.Transformer = nn.Linear(hidden_size*directions, hidden_size*directions)

        self.NTN = NTN(hidden_size*directions, hidden_size*directions, ntn_hidden)


    def forward(self, support, query, sup_len, que_len):
        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        support = support.view(n * k, sup_seq_len)

        support, query = self.EmbedDrop(self.Embedding(support)), \
                         self.EmbedDrop(self.Embedding(query))

        support, query = self.Encoder(support, sup_len), self.Encoder(query, que_len)
        support, query = self.Decoder(support, sup_len), self.Decoder(query, que_len)

        # 计算类的原型向量
        # shape: [n, k, d]
        support = support.view(n, k, -1)
        d = support.size(2)

        # coupling shape: [n, d]
        coupling = t.zeros_like(support).sum(dim=2)
        proto = None
        # 使用动态路由来计算原型向量
        for i in range(self.Iters):
            coupling, proto = dynamicRouting(self.Transformer,
                                             support, coupling,
                                             k)

        support = repeatProtoToCompShape(proto, qk, n)
        query = repeatQueryToCompShape(query, qk, n)

        return self.NTN(support, query, n).view(qk, n)






