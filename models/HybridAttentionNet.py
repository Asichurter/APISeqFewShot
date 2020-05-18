import torch as t
import torch.nn as nn
import warnings

from components.modules import CNNBlock2D
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder
from components.sequence.TCN import TemporalConvNet
from utils.training import extractTaskStructFromInput

class InstanceAttention(nn.Module):
    def __init__(self, linear_in, linear_out):
        super(InstanceAttention, self).__init__()
        self.g = nn.Linear(linear_in, linear_out)

    def forward(self, support, query, k, qk, n):
        # support/query shape: [qk*n*k, d]
        d = support.size(1)
        support = self.g(support)
        query = self.g(query)
        # shape: [qk, n, k, d]->[qk, n, k]
        attentions = t.tanh((support*query).view(qk, n, k, d)).sum(dim=3).squeeze()
        # shape: [qk,n,k]->[qk,n,k,d]
        attentions = t.softmax(attentions, dim=2).unsqueeze(3).repeat(1,1,1,d)

        return t.mul(attentions, support.view(qk, n, k, d))


class HAPNet(nn.Module):
    def __init__(self, k,
                 pretrained_matrix,
                 embed_size,
                 hidden_size=128,
                 layer_num=1,
                 self_att_dim=64,
                 word_cnt=None,
                 **modelParams):
        super(HAPNet, self).__init__()

        assert pretrained_matrix is not None or word_cnt is not None, \
            '至少需要提供词个数或者预训练的词矩阵两者一个'
        if pretrained_matrix is not None:
            self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix,
                                                          freeze=False,
                                                          padding_idx=0)
        else:
            self.Embedding = nn.Embedding(word_cnt, embed_size, padding_idx=0)


        self.EmbedNorm = nn.LayerNorm(embed_size)
        self.Encoder = BiLstmEncoder(embed_size,
                                     **modelParams)
        # self.Encoder = TemporalConvNet(**modelParams)

        # 嵌入后的向量维度
        feature_dim = 2*hidden_size#modelParams['num_channels'][-1]#

        self.CnnEncoder = CNNEncoder1D(num_channels=[feature_dim, feature_dim])

        # 获得样例注意力的模块
        # 将嵌入后的向量拼接成单通道矩阵后，有多少个支持集就为几个batch
        if k%2==0:
            warnings.warn("K=%d是偶数将会导致feature_attention中卷积核的宽度为偶数，因此部分将会发生一些变化")
            attention_paddings = [(k // 2, 0), (k // 2, 0), (0, 0)]
        else:
            attention_paddings = [(k // 2, 0), (k // 2, 0), (0, 0)]
        attention_channels = [1,32,64,1]
        attention_strides = [(1,1),(1,1),(k,1)]
        attention_kernels = [(k,1),(k,1),(k,1)]
        attention_relus = ['leaky','leaky','leaky']


        self.FeatureAttention = nn.Sequential(
            *[CNNBlock2D(attention_channels[i],
                         attention_channels[i+1],
                         attention_strides[i],
                         attention_kernels[i],
                         attention_paddings[i],
                         attention_relus[i],
                         pool=None)
              for i in range(len(attention_channels)-1)])

        # 获得样例注意力的模块
        # 将support重复query次，query重复n*k次，因为每个support在每个query下嵌入都不同
        self.InstanceAttention = InstanceAttention(feature_dim, feature_dim)


    def forward(self, support, query, sup_len, que_len):
        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)


        support = support.view(n*k, sup_seq_len)

        support = self.EmbedNorm(self.Embedding(support))
        query = self.EmbedNorm(self.Embedding(query))

        support = self.Encoder(support, sup_len)
        query = self.Encoder(query, que_len)

        support = self.CnnEncoder(support, sup_len)
        query = self.CnnEncoder(query, que_len)

        assert support.size(1)==query.size(1), '支持集维度 %d 和查询集维度 %d 必须相同!'%\
                                               (support.size(1),query.size(1))

        dim = support.size(1)

        # 将嵌入的支持集展为合适形状
        # support shape: [n,k,d]->[n,k,d]
        support = support.view(n,k,dim)
        # query shape: [qk, d]
        query = query.view(qk,-1)

        # 将支持集嵌入视为一个单通道矩阵输入到特征注意力模块中获得特征注意力
        # 并重复qk次让基于支持集的特征注意力对于qk个样本相同
        # 输入: [n,k,d]->[n,1,k,d]
        # 输出: [n,1,1,d]->[n,d]->[qk,n,d]
        feature_attentions = self.FeatureAttention(support.unsqueeze(dim=1)).squeeze().repeat(qk,1,1)

        # 将支持集重复qk次，将查询集重复n*k次以获得qk*n*k长度的样本
        # 便于在获得样例注意力时，对不同的查询集有不同的样例注意力
        # 将qk，n与k均压缩到一个维度上以便输入到线性层中
        # query_expand shape:[qk,d]->[n*k,qk,d]->[qk,n,k,d]
        # support_expand shape: [n,k,d]->[qk,n,k,d]
        support_expand = support.repeat((qk,1,1,1)).view(qk*n*k,-1)
        query_expand = query.repeat((n*k,1,1)).transpose(0,1).contiguous().view(qk*n*k,-1)

        # 利用样例注意力注意力对齐支持集样本
        # shape: [qk,n,k,d]
        support = self.InstanceAttention(support_expand, query_expand, k, qk, n)

        # 生成对于每一个qk都不同的类原型向量
        # 注意力对齐以后，将同一类内部的加权的向量相加以后
        # proto shape: [qk,n,k,d]->[qk,n,d]
        support = support.sum(dim=2).squeeze()
        # support = support.mean(dim=1).repeat((qk,1,1)).view(qk,n,-1)

        # query shape: [qk,d]->[qk,n,d]
        query = query.unsqueeze(dim=1).repeat(1,n,1)

        # dis_attented shape: [qk*n,n,d]->[qk*n,n,d]->[qk*n,n]
        # dis_attented = (((support-query)**2)).sum(dim=2).neg()
        dis_attented = (((support-query)**2)*feature_attentions).sum(dim=2).neg()

        return t.log_softmax(dis_attented, dim=1)








