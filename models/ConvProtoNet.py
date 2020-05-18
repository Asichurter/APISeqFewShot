import warnings

from components.modules import *
from components.sequence.LSTM import BiLstmEncoder
from utils.training import extractTaskStructFromInput, \
                            repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter

class ConvProtoNet(nn.Module):
    def __init__(self, k,
                 pretrained_matrix,
                 embed_size,
                 hidden_size=128,
                 layer_num=1,
                 self_att_dim=64,
                 word_cnt=None):
        super(ConvProtoNet, self).__init__()

        # 可训练的嵌入层
        if pretrained_matrix is not None:
            self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        else:
            self.Embedding = nn.Embedding(word_cnt, embedding_dim=embed_size, padding_idx=0)

        self.EmbedNorm = nn.LayerNorm(embed_size)

        # self.Encoder = CNNEncoder2D(dims=[1, 64, 128, 256, 256],
        #                             kernel_sizes=[3,3,3,3],
        #                             paddings=[1,1,1,1],
        #                             relus=[True,True,True,True],
        #                             pools=['max','max','max','ada'])
        # self.Encoder = CNNEncoder1D(dims=[embed_size, 64, 128, 256, 256],
        #                             kernel_sizes=[3,3,3,3],
        #                             paddings=[1,1,1,1],
        #                             relus=[True,True,True,True],
        #                             pools=['max','max','max','ada'])
        # self.Encoder = CNNEncoder1D(dims=[embed_size, 256, 256],
        #                             kernel_sizes=[3,3],
        #                             paddings=[1,1],
        #                             relus=[True,True],
        #                             pools=['max','ada'])

        # self.Encoder = TransformerEncoder(layer_num=layer_num,
        #                                   embedding_size=embed_size,
        #                                   feature_size=hidden,
        #                                   att_hid=self_att_dim,
        #                                   reduce=False)
        self.Encoder = BiLstmEncoder(embed_size,  #64
                                     hidden_size=hidden_size,
                                     layer_num=layer_num,
                                     self_att_dim=self_att_dim,
                                     useBN=False)
        # self.Encoder = BiLstmCellEncoder(input_size=embed_size,
        #                                  hidden_size=hidden,
        #                                  num_layers=layer_num,
        #                                  bidirectional=True,
        #                                  self_att_dim=self_att_dim)

        # self.CNN = CNNEncoder1D(dims=[hidden_size * 2, 512])
        # self.CNN = CnnNGramEncoder(dims=[1,32,64],
        #                            kernel_sizes=[(3,embed_size),(3,embed_size//2+1)],
        #                            paddings=[(1,embed_size//4),(1,embed_size//8)],
        #                            relus=[True,True])

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
        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(n*k, sup_seq_len)

        # shape: [batch, seq, dim]
        support = self.Embedding(support)
        query = self.Embedding(query)

        support = self.EmbedNorm(support)
        query = self.EmbedNorm(query)

        # support = self.CNN(support)
        # query = self.CNN(query)

        # # # pack以便输入到LSTM中
        # support = pack_padded_sequence(support, sup_len, batch_first=True, enforce_sorted=False)
        # query = pack_padded_sequence(query, que_len, batch_first=True, enforce_sorted=False)

        # shape: [batch, dim]
        support = self.Encoder(support, sup_len)
        query = self.Encoder(query, que_len)

        # support, sup_len = pad_packed_sequence(support, batch_first=True)
        # query, que_len = pad_packed_sequence(query, batch_first=True)

        # support = avgOverHiddenStates(support, sup_len)
        # query = avgOverHiddenStates(query, que_len)

        # support = self.CNN(support, sup_len)
        # query = self.CNN(query, que_len)

        # support, s_len = pad_packed_sequence(support, batch_first=True, enforce_sorted=False)
        # query, q_len = pad_packed_sequence(query, batch_first=True, enforce_sorted=False)

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




