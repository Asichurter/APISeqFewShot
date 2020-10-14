import logging

from components.modules import *
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder, BiLstmCellEncoder
from utils.training import extractTaskStructFromInput


class FT(nn.Module):
    def __init__(self, n, loss_fn,
                 pretrained_matrix,
                 embed_size,
                 word_cnt=None,
                 lr = 0.01,
                 **modelParams):
        super(FT, self).__init__()

        self.Lr = lr
        self.LossFn = loss_fn
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

        self.Decoder = CNNEncoder1D([hidden_size,hidden_size])

        self.Classifier = nn.Linear(hidden_size, n)


    def _embed(self, x, lens):
        x = self.EmbedDrop(self.Embedding(x))
        x = self.Encoder(x, lens)
        if self.MiddleEncoder is not None:
            x = self.MiddleEncoder(x, lens)
        x = self.Decoder(x, lens)

        return x

    def forward(self, seqs, lens):
        seqs = self._embed(seqs, lens)
        preds = self.Classifier(seqs)

        return F.log_softmax(preds, dim=1)


    # def forward(self, support, query, sup_len, que_len, sup_labels):
    #     if self.DataParallel:
    #         support = support.squeeze(0)
    #         sup_len = sup_len[0]
    #
    #     n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)
    #     classifier = nn.Linear(self.HiddenSize, n).cuda()
    #
    #     # 提取了任务结构后，将所有样本展平为一个批次
    #     support = support.view(n*k, sup_seq_len)
    #
    #     support, query = self._embed(support, sup_len), \
    #                      self._embed(query, que_len)
    #
    #
    #     assert support.size(1)==query.size(1), '支持集维度 %d 和查询集维度 %d 必须相同!'%\
    #                                            (support.size(1),query.size(1))
    #
    #     for i in range(5):
    #         sup_preds = t.softmax(classifier(support), dim=1)
    #         loss_fn = self.LossFn
    #
    #         loss_val = loss_fn(sup_preds, sup_labels)
    #         grads = t.autograd.grad(loss_val, classifier.parameters(), create_graph=True)
    #         state_dict = {
    #             key: val.clone()
    #             for key,val in classifier.state_dict().items()
    #         }
    #
    #         for g,(key,val) in zip(grads, state_dict.items()):
    #             val -= self.Lr * g
    #
    #         classifier.load_state_dict(state_dict)
    #
    #     preds = t.softmax(classifier(query), dim=1)
    #     return preds




