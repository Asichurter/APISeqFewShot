import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from components.sequence.FastText import FastTextEncoder
from components.set2set.deepset import DeepSet
from components.set2set.transformer import TransformerSet
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder
from components.reduction.selfatt import AttnReduction
from components.reduction.max import StepMaxReduce
from components.sequence.TCN import TemporalConvNet
from utils.training import extractTaskStructFromInput, repeatProtoToCompShape, repeatQueryToCompShape, protoDisAdapter

class FEAT(nn.Module):

    def __init__(self,
                 pretrained_matrix,
                 embed_size,
                 feat_avg='pre',
                 contrastive_factor=None,
                 **modelParams):

        super(FEAT, self).__init__()

        self.DataParallel = modelParams['data_parallel']

        self.Avg = feat_avg
        self.ContraFac = contrastive_factor
        self.DisTempr = modelParams['temperature'] if 'temperature' in modelParams else 1

        # self.Encoder = FastTextEncoder(pretrained_matrix,
        #                                embed_size,
        #                                modelParams['dropout'])

        # 可训练的嵌入层
        self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        self.EmbedDrop = nn.Dropout(modelParams['dropout'])
        self.EmbedNorm = nn.LayerNorm(embed_size)
        #
        self.Encoder = BiLstmEncoder(input_size=embed_size,
                                     **modelParams)

        # self.Encoder = TemporalConvNet(num_inputs=embed_size,
        #                                init_hidden_channel=modelParams['tcn_init_channel'],
        #                                num_channels=modelParams['tcn_channels'])

        self.Decoder = CNNEncoder1D([(modelParams['bidirectional']+1)*modelParams['hidden_size'],
                                     (modelParams['bidirectional']+1)*modelParams['hidden_size']])
        # self.Decoder = StepMaxReduce()

        if modelParams['set_function'] == 'deepset':
            self.SetFunc = DeepSet(embed_dim=(modelParams['bidirectional']+1)*modelParams['hidden_size'],
                                   **modelParams)
        elif modelParams['set_function'] == 'transformer':
            self.SetFunc = TransformerSet(trans_input_size=(modelParams['bidirectional']+1)*modelParams['hidden_size'],
                                          **modelParams)
        else:
            raise ValueError('Unrecognized set function type:', modelParams['set_function'])


    def forward(self, support, query, sup_len, que_len,
                metric='euc', return_unadapted=False):

        if self.DataParallel:
            support = support.squeeze(0)
            sup_len = sup_len[0]

        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        qk_per_class = qk // n

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(n*k, sup_seq_len)

        # ------------------------------------------------------
        # shape: [batch, seq, dim]
        support = self.Embedding(support)
        query = self.Embedding(query)

        support = self.EmbedDrop(support)
        query = self.EmbedDrop(query)

        # shape: [batch, dim]
        support = self.Encoder(support, sup_len)
        query = self.Encoder(query, que_len)

        support = self.Decoder(support, sup_len)
        query = self.Decoder(query, que_len)
        # ------------------------------------------------------

        # support = self.Encoder(support, sup_len)
        # query = self.Encoder(query, que_len)

        assert support.size(1)==query.size(1), '支持集维度 %d 和查询集维度 %d 必须相同!'%\
                                               (support.size(1),query.size(1))

        dim = support.size(1)

        # contrastive-loss for regulization during training
        if self.training and self.ContraFac is not None:
            # union shape: [n, qk+k, dim]
            # here suppose query set is constructed in group by class
            union = t.cat((support.view(n, k, dim), query.view(n, qk_per_class, dim)),
                          dim=1)                        # TODO: make it capable to process in batch

            adapted_union = self.SetFunc(union)

            # post-avg in default
            adapted_proto = adapted_union.mean(dim=1)

            # union shape: [(qk+k)*n, dim]
            adapted_union = adapted_union.view((qk_per_class + k) * n, dim)

            # let the whole dataset execute classification task based on the adapted prototypes
            adapted_proto = repeatProtoToCompShape(adapted_proto, (qk_per_class + k) * n, n)
            adapted_union = repeatQueryToCompShape(adapted_union, (qk_per_class + k) * n, n)

            adapted_sim = protoDisAdapter(adapted_proto, adapted_union,
                                          (qk_per_class + k) * n, n, dim, dis_type='euc')

            # here, the target label set has labels for both support set and query set,
            # where labels permute in order and cluster (every 'qk_per_class+k')
            adapted_res = F.log_softmax(adapted_sim, dim=1)

        if return_unadapted:
            unada_support = support.view(n,k,-1).mean(1)
            unada_support = repeatProtoToCompShape(unada_support,
                                                   qk, n)

        ################################################################
        if self.Avg == 'post':

            # support set2set
            support = self.SetFunc(support.view(1,n*k,dim))

            # shape: [n, dim]
            support = support.view(n, k, dim).mean(dim=1)

        elif self.Avg == 'pre':

            # shape: [n, dim]
            support = support.view(n, k, dim).mean(dim=1)
            # support set2set
            support = self.SetFunc(support.unsqueeze(0))
        ################################################################


        # shape: [n, dim] -> [1, n, dim]
        # pre-avg in default, treat prototypes as sequence
        # support = support.view(n, k, dim).mean(dim=1).unsqueeze(0)
        # # support set2set
        # support = self.SetFunc(support)

        support = repeatProtoToCompShape(support, qk, n)
        query = repeatQueryToCompShape(query, qk, n)

        similarity = protoDisAdapter(support, query, qk, n, dim,
                                     dis_type='euc',
                                     temperature=self.DisTempr)

        if self.training and self.ContraFac is not None:
            return F.log_softmax(similarity, dim=1), adapted_res

        else:
            if return_unadapted:
                unada_sim = protoDisAdapter(unada_support, query, qk, n, dim,
                                     dis_type='euc',
                                     temperature=self.DisTempr)
                return F.log_softmax(similarity, dim=1), F.log_softmax(unada_sim, dim=1)

            else:
                return F.log_softmax(similarity, dim=1)

