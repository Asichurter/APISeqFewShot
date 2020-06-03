import logging

from components.modules import *
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder
from utils.training import extractTaskStructFromInput, \
                            repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter


class MatchNet(nn.Module):
    def __init__(self, pretrained_matrix,
                 embed_size,
                 **modelParams):
        super(MatchNet, self).__init__()

        # 可训练的嵌入层
        self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        self.EmbedNorm = nn.LayerNorm(embed_size)
        self.EmbedDrop = nn.Dropout(modelParams['dropout'])

        self.Encoder = BiLstmEncoder(input_size=embed_size, **modelParams)

        self.Decoder = CNNEncoder1D([modelParams['hidden_size'],
                                     modelParams['hidden_size']])

    def forward(self, support, query, sup_len, que_len, metric='euc'):
        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(n*k, sup_seq_len)

        # shape: [batch, seq, dim]
        support = self.Embedding(support)
        query = self.Embedding(query)

        support = self.EmbedDrop(self.EmbedNorm(support))
        query = self.EmbedDrop(self.EmbedNorm(query))

        # shape: [batch, dim]
        support = self.Encoder(support, sup_len)
        query = self.Encoder(query, que_len)


        support = self.Decoder(support, sup_len)
        query = self.Decoder(query, que_len)

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