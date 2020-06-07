import warnings

from components.modules import *
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder
from utils.training import extractTaskStructFromInput, \
                            repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter

class ConvProtoNet(nn.Module):
    def __init__(self, k,
                 pretrained_matrix,
                 embed_size,
                 **modelParams):
        super(ConvProtoNet, self).__init__()

        self.DataParallel = modelParams['data_parallel']

        # 可训练的嵌入层
        self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        self.EmbedDrop = nn.Dropout(modelParams['dropout'])
        # self.EmbedNorm = nn.LayerNorm(embed_size)
        #
        self.Encoder = BiLstmEncoder(input_size=embed_size,
                                     **modelParams)

        # self.Encoder = TemporalConvNet(num_inputs=embed_size,
        #                                init_hidden_channel=modelParams['tcn_init_channel'],
        #                                num_channels=modelParams['tcn_channels'])

        self.Decoder = CNNEncoder1D([(modelParams['bidirectional']+1)*modelParams['hidden_size'],
                                     (modelParams['bidirectional']+1)*modelParams['hidden_size']])


        if k%2==0:
            warnings.warn("K=%d是偶数将会导致feature_attention中卷积核的宽度为偶数，因此部分将会发生一些变化")
            attention_paddings = [(k // 2, 0), (k // 2, 0), (0, 0)]
        else:
            attention_paddings = [(k // 2, 0), (k // 2, 0), (0, 0)]
        attention_channels = [1,32,64,1]
        attention_strides = [(1,1),(1,1),(k,1)]
        attention_kernels = [(k,1),(k,1),(k,1)]
        attention_relus = ['relu','relu',None]
        self.Induction = nn.Sequential(
            *[CNNBlock2D(attention_channels[i],
                         attention_channels[i + 1],
                         attention_strides[i],
                         attention_kernels[i],
                         attention_paddings[i],
                         attention_relus[i],
                         pool=None)
              for i in range(len(attention_channels) - 1)]
        )

    def forward(self, support, query, sup_len, que_len, metric='euc'):

        if self.DataParallel:
            support = support.squeeze(0)
            sup_len = sup_len[0]

        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(n*k, sup_seq_len)

        # ------------------------------------------------------
        # shape: [batch, seq, dim]
        support = self.EmbedDrop(self.Embedding(support))
        query = self.EmbedDrop(self.Embedding(query))

        # support = self.EmbedDrop(self.EmbedNorm(support))
        # query = self.EmbedDrop(self.EmbedNorm(query))

        # shape: [batch, dim]
        support = self.Encoder(support, sup_len)
        query = self.Encoder(query, que_len)

        support = self.Decoder(support, sup_len)
        query = self.Decoder(query, que_len)
        # ------------------------------------------------------

        assert support.size(1)==query.size(1), '支持集维度 %d 和查询集维度 %d 必须相同!'%\
                                               (support.size(1),query.size(1))

        dim = support.size(1)

        # 原型向量
        # shape: [n, dim]
        support = support.view(n, k, dim)
        support = self.Induction(support.unsqueeze(1)).squeeze()

        # 整型成为可比较的形状: [qk, n, dim]
        support = repeatProtoToCompShape(support, qk, n)
        query = repeatQueryToCompShape(query, qk, n)

        similarity = protoDisAdapter(support, query, qk, n, dim, dis_type='cos')

        # return t.softmax(similarity, dim=1)
        return F.log_softmax(similarity, dim=1)




