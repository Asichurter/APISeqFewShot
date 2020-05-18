from components.modules import *
from components.sequence.CNN import CNNEncoder1D
from utils.training import extractTaskStructFromInput, \
                            repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter


class TCProtoNet(nn.Module):
    def __init__(self, pretrained_matrix,
                 embed_size,
                 hidden=128,
                 layer_num=1,
                 self_att_dim=64,
                 word_cnt=None,
                 **modelParams):
        super(TCProtoNet, self).__init__()

        # 可训练的嵌入层
        if pretrained_matrix is not None:
            self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        else:
            self.Embedding = nn.Embedding(word_cnt, embedding_dim=embed_size, padding_idx=0)

        self.EmbedNorm = nn.LayerNorm(embed_size)

        self.Encoder = CNNEncoder1D(**modelParams)

        # self.Encoder = TransformerEncoder(layer_num=layer_num,
        #                                   embedding_size=embed_size,
        #                                   feature_size=hidden,
        #                                   att_hid=self_att_dim,
        #                                   reduce=False)
        # self.Encoder = BiLstmEncoder(embed_size,#64
        #                              hidden_size=hidden,
        #                              layer_num=layer_num,
        #                              self_att_dim=self_att_dim,
        #                              useBN=False)

        # self.Encoder = TemporalConvNet(**modelParams)

        # self.TEN = TenStepAffine1D(task_dim=2*hidden, step_length=50)       # seq len fix to 50
        self.TEN = TenStepAffine1D(task_dim=modelParams['num_channels'][-1],
                                   feature_dim=modelParams['num_channels'][-1])


        # self.CNN = CNNEncoder1D(dims=[hidden*2, hidden*2])
        self.CNN = CNNEncoder1D(num_channels=[modelParams['num_channels'][-1],
                                              modelParams['num_channels'][-1]])


    def forward(self, support, query, sup_len, que_len, metric='euc'):
        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)


        # forehead forward to obtain task prototype
        f_support = support.view(n * k, sup_seq_len)
        f_support = self.Embedding(f_support)
        f_support = self.EmbedNorm(f_support)
        f_support = self.Encoder(f_support, sup_len)
        f_support = self.CNN(f_support, sup_len)

        f_support = f_support.view(n, k, -1)

        task_proto = f_support.mean((0,1))


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

        # task-conditioning affine
        support = self.TEN(support, task_proto)
        query = self.TEN(query, task_proto)

        support = self.CNN(support, sup_len)
        query = self.CNN(query, que_len)

        assert support.size(1)==query.size(1), '支持集维度 %d 和查询集维度 %d 必须相同!'%\
                                               (support.size(1),query.size(1))
        dim = support.size(1)

        # 原型向量
        # shape: [n, dim]
        support = support.view(n, k, dim).mean(dim=1)

        # 整型成为可比较的形状: [qk, n, dim]
        support = repeatProtoToCompShape(support, qk, n)
        query = repeatQueryToCompShape(query, qk, n)

        similarity = protoDisAdapter(support, query, qk, n, dim, dis_type='euc')

        # return t.softmax(similarity, dim=1)
        return F.log_softmax(similarity, dim=1)
        # return t.sigmoid(similarity)

    def penalizedNorm(self):
        w_norm, b_norm = self.TEN.penalizedNorm()
        return w_norm+b_norm


