import logging

from components.modules import *
from components.reduction.selfatt import BiliAttnReduction, SelfAttnReduction
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder, BiLstmCellEncoder
from components.reduction.max import StepMaxReduce
from components.sequence.TCN import TemporalConvNet
from components.sequence.transformer import TransformerEncoder, MultiHeadAttention
from utils.training import extractTaskStructFromInput, \
                            repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter


class ProtoNet(nn.Module):
    def __init__(self, pretrained_matrix,
                 embed_size,
                 word_cnt=None,
                 **modelParams):
        super(ProtoNet, self).__init__()

        self.DistTemp = modelParams['temperature'] if 'temperature' in modelParams else 1
        self.DataParallel = modelParams['data_parallel'] if 'data_parallel' in modelParams else False

        # 可训练的嵌入层
        if pretrained_matrix is not None:
            self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        else:
            self.Embedding = nn.Embedding(word_cnt, embedding_dim=embed_size, padding_idx=0)

        # self.EmbedNorm = nn.LayerNorm(embed_size)
        self.EmbedDrop = nn.Dropout(modelParams['dropout'])

        hidden_size = (1 + modelParams['bidirectional']) * modelParams['hidden_size']

        #------------------------------------------------------------------------
        self.Encoder = BiLstmEncoder(input_size=embed_size, **modelParams)
        # self.Encoder = BiLstmCellEncoder(input_size=embed_size, **modelParams)
        #------------------------------------------------------------------------

        #------------------------------------------------------------------------
        self.MiddleEncoder = None#MultiHeadAttention(mhatt_input_size=hidden_size, **modelParams)
        #------------------------------------------------------------------------

        # self.Encoder = TransformerEncoder(embed_size=embed_size, **modelParams)

        # self.Encoder = CNNEncoder2D(dims=[1, 64, 128, 256, 256],
        #                             kernel_sizes=[3,3,3,3],
        #                             paddings=[1,1,1,1],
        #                             relus=[True,True,True,True],
        #                             pools=['max','max','max','ada'])
        # self.Encoder = CNNEncoder1D(**modelParams)
        # self.Encoder = CNNEncoder1D(**kwargs)

        # self.Encoder =  BiLstmEncoder(embed_size,  # 64
        #                               hidden_size=hidden,
        #                               layer_num=layer_num,
        #                               self_att_dim=self_att_dim,
        #                               useBN=False)
        # self.Encoder = TemporalConvNet(**modelParams)

        # self.Encoder = nn.ModuleList([
        #     BiLstmEncoder(embed_size,  # 64
        #                   hidden_size=hidden,
        #                   layer_num=1,
        #                   self_att_dim=self_att_dim,
        #                   useBN=False),
        #     BiLstmEncoder(2*hidden,  # 64
        #                   hidden_size=hidden,
        #                   layer_num=1,
        #                   self_att_dim=self_att_dim,
        #                   useBN=False)
        # ])
        # self.EncoderNorm = nn.ModuleList([
        #     nn.LayerNorm(2*hidden),
        #     nn.LayerNorm(2*hidden)
        # ])

        # self.Decoder = StepMaxReduce()

        # self.Encoder = BiLstmCellEncoder(input_size=embed_size,
        #                                  hidden_size=hidden,
        #                                  num_layers=layer_num,
        #                                  bidirectional=True,
        #                                  self_att_dim=self_att_dim)

        self.Decoder = CNNEncoder1D([hidden_size,hidden_size])
        # self.Decoder = AttnReduction(input_dim=hidden_size, hidden_dim=hidden_size)
        # self.Decoder = SelfAttnReduction(input_size=hidden_size, **modelParams)
        # self.Reduce = CNNEncoder1D([modelParams['num_channels'][-1],
        #                             modelParams['num_channels'][-1]])
        # self.Reduce = StepMaxReduce()

        # self.CNN = CnnNGramEncoder(dims=[1,32,64],
        #                            kernel_sizes=[(3,embed_size),(3,embed_size//2+1)],
        #                            paddings=[(1,embed_size//4),(1,embed_size//8)],
        #                            relus=[True,True])


    def _embed(self, x, lens):
        x = self.EmbedDrop(self.Embedding(x))
        x = self.Encoder(x, lens)
        if self.MiddleEncoder is not None:
            x = self.MiddleEncoder(x, lens)
        x = self.Decoder(x, lens)

        return x

    def forward(self, support, query, sup_len, que_len, metric='euc', return_embeddings=False):

        if self.DataParallel:
            support = support.squeeze(0)
            sup_len = sup_len[0]

        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(n*k, sup_seq_len)

        support, query = self._embed(support, sup_len), \
                         self._embed(query, que_len)


        assert support.size(1)==query.size(1), '支持集维度 %d 和查询集维度 %d 必须相同!'%\
                                               (support.size(1),query.size(1))

        dim = support.size(1)

        # 原型向量
        # shape: [n, dim]
        orig_protos = support.view(n, k, dim).mean(dim=1)

        # 整型成为可比较的形状: [qk, n, dim]
        protos = repeatProtoToCompShape(orig_protos, qk, n)
        rep_query = repeatQueryToCompShape(query, qk, n)

        similarity = protoDisAdapter(protos, rep_query, qk, n, dim,
                                     dis_type='euc',
                                     temperature=self.DistTemp)

        # return t.softmax(similarity, dim=1)
        if return_embeddings:
            return support, query.view(qk,-1), orig_protos, F.log_softmax(similarity, dim=1)
        return F.log_softmax(similarity, dim=1)





















class IncepProtoNet(nn.Module):

    def __init__(self, channels, depth=3):
        super(IncepProtoNet, self).__init__()

        if depth%2 != 1:
            logging.warning('3D卷积深度为偶数，会导致因为无法padding而序列长度发生变化！')

        layers = [ResInception(in_channel=channels[i] if i==0 else 4*channels[i],
                               out_channel=channels[i+1],
                               depth=depth)
                  for i in range(len(channels)-1)]

        self.Encoder = nn.Sequential(*layers)

        # TODO: 添加多个Inception模块后的Pool

    def forward(self, support, query, sup_len=None, que_len=None):
        # input shape:
        # sup=[n, k, 1, sup_seq_len, height, width]
        # que=[batch, 1, que_seq_len, height, width]
        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query,
                                                                        unsqueezed=True,
                                                                        is_matrix=True)
        height, width = query.size(3), query.size(4)

        support = support.view(n*k, 1, sup_seq_len, height, width)
        query = query.view(qk, 1, que_seq_len, height, width)

        # output shape: [batch, out_channel=1, seq_len, height, width]
        support = self.Encoder(support).squeeze()
        query = self.Encoder(query).squeeze()

        # TODO:直接展开序列作为特征
        support = support.view(n, k, -1).mean(dim=1)
        query = query.view(qk, -1)

        assert support.size(1)==query.size(1), \
            '支持集和查询集的嵌入后特征维度不相同！'

        dim = support.size(1)

        # 整型成为可比较的形状: [qk, n, dim]
        support = repeatProtoToCompShape(support, qk, n)
        query = repeatQueryToCompShape(query, qk, n)

        similarity = protoDisAdapter(support, query, qk, n, dim, dis_type='cos')

        # return t.softmax(similarity, dim=1)
        return F.log_softmax(similarity, dim=1)

##########################################################
# 先使用3D卷积来提取每个时间步中的矩阵特征，再将时间步提取到的特征输入
# 到LSTM中，使用注意力机制从序列中归纳表示
##########################################################
class CNNLstmProtoNet(nn.Module):

    def  __init__(self,
                 channels=[1,32,64,64],     # 默认3个卷积层
                 lstm_input_size=64*2*2,    # matrix大小为10×10，两次池化为2×2
                 strides=None,
                 hidden_size=64,
                 layer_num=1,
                 self_att_dim=32):
        super(CNNLstmProtoNet, self).__init__()

        self.Embedding = CNNEncoder(channels=channels,
                                    strides=strides,
                                    flatten=False,      # 保留序列信息
                                    pools=[True,True,False]
                                    )

        self.LstmEncoder = BiLstmEncoder(input_size=lstm_input_size,
                                         hidden_size=hidden_size,
                                         layer_num=layer_num,
                                         self_att_dim=self_att_dim)

    def forward(self, support, query, sup_len=None, que_len=None):
        # input shape:
        # sup=[n, k, sup_seq_len, height, width]
        # que=[qk, que_seq_len, height, width]
        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query,
                                                                        unsqueezed=False,
                                                                        is_matrix=True)
        height, width = query.size(2), query.size(3)

        support = support.view(n*k, sup_seq_len, height, width)

        # output shape: [batch, seq_len, feature]
        support = self.Embedding(support)
        query = self.Embedding(query)

        support = self.LstmEncoder(support)
        query = self.LstmEncoder(query)

        # TODO:直接展开序列作为特征
        support = support.view(n, k, -1).mean(dim=1)
        query = query.view(qk, -1)

        assert support.size(1)==query.size(1), \
            '支持集和查询集的嵌入后特征维度不相同！'

        dim = support.size(1)

        # 整型成为可比较的形状: [qk, n, dim]
        support = repeatProtoToCompShape(support, qk, n)
        query = repeatQueryToCompShape(query, qk, n)

        similarity = protoDisAdapter(support, query, qk, n, dim, dis_type='cos')

        # return t.softmax(similarity, dim=1)
        return F.log_softmax(similarity, dim=1)


if __name__ == '__main__':
    n = ProtoNet(None, 64, word_cnt=100)

















































































class ImageProtoNet(nn.Module):
    def __init__(self, in_channels=1, metric="SqEuc", **kwargs):
        super(ImageProtoNet, self).__init__()

        self.metric = metric
        self.ProtoNorm = None
        self.inChanel = in_channels

        # 第一层是一个1输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 经过这层以后，尺寸除以4

        # TODO: stride已修改
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第二层是一个64输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        # 卷积核的宽度为3,13变为10，再经过宽度为2的pool变为5
        # 经过这层以后，尺寸除以4
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第三层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数,的卷积层
        # 经过这层以后，尺寸除以2
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # 第四层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数的卷积层
        # 经过这层以后，尺寸除以2
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)#SppPooling(levels=[1,2])
        )
        # self.Transformer = nn.Linear(256, 256, bias=False)

    def forward(self, support, query, save_embed=False, save_proto=False):
        assert len(support.size()) == 5 and len(query.size()) == 4, \
            "support必须遵循(n,k,c,w,w)的格式，query必须遵循(l,c,w,w)的格式！"
        k = support.size(1)
        qk = query.size(0)
        n = support.size(0)
        w = support.size(3)

        support = support.view(n*k, self.inChanel, w, w)
        query = query.view(qk, self.inChanel, w, w)

        # input shape: [n, k, d]
        def proto_mean(tensors):
            return tensors.mean(dim=1).squeeze()

        # 每一层都是以上一层的输出为输入，得到新的输出、
        # 支持集输入是N个类，每个类有K个实例
        support = self.layer1(support)
        support = self.layer2(support)
        support = self.layer3(support)
        support = self.layer4(support).squeeze()
        # support = self.Layers(support).squeeze()

        # 查询集的输入是N个类，每个类有qk个实例
        # 但是，测试的时候也需要支持单样本的查询
        query = self.layer1(query)
        query = self.layer2(query)
        query = self.layer3(query)
        query = self.layer4(query).squeeze().view(qk,-1)

        # 计算类的原型向量
        # shape: [n, k, d]
        support = support.view(n, k, -1)

        proto = proto_mean(support)
        support = proto

        # 将原型向量与查询集打包
        # shape: [n,d]->[qk, n, d]
        support = support.repeat((qk,1,1)).view(qk,n,-1)

        # query shape: [qk,d]->[n,qk,d]->[qk,n,d]
        query = query.repeat(n,1,1).transpose(0,1).contiguous().view(qk,n,-1)

        if self.metric == "SqEuc":
            # 由于pytorch中的NLLLoss需要接受对数概率，根据官网上的提示最后一层改为log_softmax
            # 已修正：以负距离输入到softmax中,而不是距离
            posterior = F.log_softmax(t.sum((query-support)**2, dim=2).sqrt().neg(),dim=1)

        elif self.metric == 'cos':
            # 在原cos相似度的基础上添加放大系数
            scale = 10
            posterior = F.cosine_similarity(query, support, dim=2)*scale
            posterior = F.log_softmax(posterior, dim=1)

        return posterior


if __name__ == '__main__':
    n = ProtoNet(None, 64, word_cnt=100)



