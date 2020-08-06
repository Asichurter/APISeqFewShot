import logging

from components.modules import *
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder, BiLstmCellEncoder
from utils.training import extractTaskStructFromInput, \
                            repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter


class NnNet(nn.Module):
    def __init__(self, pretrained_matrix,
                 embed_size,
                 **modelParams):
        super(NnNet, self).__init__()

        self.DataParallel = modelParams['data_parallel'] if 'data_parallel' in modelParams else False

        # 可训练的嵌入层
        self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        # self.EmbedNorm = nn.LayerNorm(embed_size)
        self.EmbedDrop = nn.Dropout(modelParams['dropout'])

        hidden_size = (1 + modelParams['bidirectional']) * modelParams['hidden_size']

        self.Encoder = BiLstmEncoder(input_size=embed_size, **modelParams)
        # self.Encoder = BiLstmCellEncoder(input_size=embed_size, **modelParams)

        self.Decoder = CNNEncoder1D([hidden_size,hidden_size])

    def _embed(self, x, lens):
        x = self.EmbedDrop(self.Embedding(x))
        x = self.Encoder(x, lens)
        x = self.Decoder(x, lens)
        return x

    def forward(self, support, query, sup_len, que_len, metric='euc'):

        if self.DataParallel:
            support = support.squeeze(0)
            sup_len = sup_len[0]

        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(n*k, sup_seq_len)

        support = self._embed(support, sup_len)
        query = self._embed(query, que_len)

        assert support.size(1)==query.size(1), '支持集维度 %d 和查询集维度 %d 必须相同!'%\
                                               (support.size(1),query.size(1))

        dim = support.size(1)

        # 整型成为可比较的形状: [qk, n, dim]
        support = support.repeat((qk,1,1)).view(qk,n*k,-1)
        query = query.repeat(n*k,1,1).transpose(0,1).contiguous().view(qk,n*k,-1)

        # directly compare with support samples, instead of prototypes
        # shape: [qk, n*k, dim]->[qk, n, k, dim] -> [qk, n]
        similarity = ((support - query) ** 2).neg().view(qk,n,k,-1).sum(-1)
        # similarity = ((support - query) ** 2).neg().view(qk,n,k,-1).sum((2,3))

        # take the closest point in each class to make comparison
        similarity = t.max(similarity, dim=2).values

        return F.log_softmax(similarity, dim=1)


if __name__ == '__main__':
    pass
    # n = ProtoNet(None, 64, word_cnt=100)