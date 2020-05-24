import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from components.set2set.deepaffine import DeepAffine
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder
from utils.training import extractTaskStructFromInput, repeatProtoToCompShape, repeatQueryToCompShape, protoDisAdapter

class AFEAT(nn.Module):

    def __init__(self,
                 pretrained_matrix,
                 embed_size,
                 feat_avg='pre',
                 contrastive_factor=None,
                 **modelParams):

        super(AFEAT, self).__init__()

        self.Avg = feat_avg
        self.ContraFac = contrastive_factor
        self.DisTempr = modelParams['temperature'] if 'temperature' in modelParams else 1

        # 可训练的嵌入层
        self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        self.EmbedNorm = nn.LayerNorm(embed_size)

        self.Encoder = BiLstmEncoder(input_size=embed_size,
                                     **modelParams)

        self.Decoder = CNNEncoder1D([(modelParams['bidirectional']+1)*modelParams['hidden_size'],
                                     (modelParams['bidirectional']+1)*modelParams['hidden_size']])

        self.SetFunc = DeepAffine(embed_dim=(modelParams['bidirectional']+1)*modelParams['hidden_size'],
                               dropout=modelParams['dropout'])

    def forward(self, support, query, sup_len, que_len, metric='euc'):
        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(n*k, sup_seq_len)

        # shape: [batch, seq, dim]
        support = self.Embedding(support)
        query = self.Embedding(query)

        support = self.EmbedNorm(support)
        query = self.EmbedNorm(query)

        # shape: [batch, dim]
        support = self.Encoder(support, sup_len)
        query = self.Encoder(query, que_len)

        support = self.Decoder(support, sup_len)
        query = self.Decoder(query, que_len)

        assert support.size(1)==query.size(1), '支持集维度 %d 和查询集维度 %d 必须相同!'%\
                                               (support.size(1),query.size(1))

        dim = support.size(1)

        # support set2set
        support_weight = t.softmax(self.SetFunc(support.view(1,n*k,dim)).view(n,k), dim=1)
        support_weight = support_weight.unsqueeze(-1).repeat(1,1,dim)

        # shape: [n, k, dim] -> [n, dim]
        support = support.view(n, k, dim)
        support = (support * support_weight).sum(dim=1)

        support = repeatProtoToCompShape(support, qk, n)
        query = repeatQueryToCompShape(query, qk, n)

        similarity = protoDisAdapter(support, query, qk, n, dim,
                                     dis_type='euc',
                                     temperature=self.DisTempr)

        return F.log_softmax(similarity, dim=1)

