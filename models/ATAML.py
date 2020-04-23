import torch as t
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn
from torch.optim.adam import Adam

from components.modules import BiLstmEncoder, CNNEncoder1D
from utils.training import extractTaskStructFromInput

def rename(name, token='-'):
    '''
    由于ParameterDict中不允许有.字符的存在，因此利用本方法来中转，将
    .字符转化为-来避开
    '''
    return name.replace('.',token)

class BaseLearner(nn.Module):
    def __init__(self, n, pretrained_matrix, embed_size, seq_len, **kwargs):
        super(BaseLearner, self).__init__()
        self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix,
                                                      freeze=False,
                                                      padding_idx=0)
        self.EmbedNorm = nn.LayerNorm(embed_size)
        self.Encoder = CNNEncoder1D(**kwargs)#BiLstmEncoder(input_size=embed_size, **kwargs)
        self.Attention = nn.Linear(seq_len, 1)

        # out_size = kwargs['hidden_size']
        self.fc = nn.Linear(kwargs['dims'][-1], n)  # 对于双向lstm，输出维度是隐藏层的两倍
                                                    # 对于CNN，输出维度是嵌入维度

    def forward(self, x, lens, params=None):
        length = x.size(0)
        x = self.Embedding(x)
        x = self.EmbedNorm(x)

        if params is None:
            # shape: [batch, seq, dim] => [batch, channel, seq_len]
            x = self.Encoder(x, lens)#.view(length, -1)
            att_weight = self.Attention(x).repeat((1,1,))

            x = self.fc(x)
        else:
            # x = F.embedding(x,
            #                 weight=params['Embedding.weight'],
            #                 padding_idx=0)
            # x = F.layer_norm(x,
            #                  normalized_shape=(params['Embedding.weight'].size(1),),
            #                  weight=params['EmbedNorm.weight'],
            #                  bias=params['EmbedNorm.bias'])
            # 使用适应后的参数来前馈
            x = self.Encoder(x, lens, params)
            x = x.view(length, -1)
            x = F.linear(
                x,
                params['fc.weight'],
                params['fc.bias'],
            )

        return F.log_softmax(x, dim=1)

    def clone_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict

class ATAML(nn.Module):
    def __init__(self, n, loss_fn, lr=1e-3, **kwargs):
        super(ATAML, self).__init__()

        #######################################################
        # For CNN only
        kwargs['dims'] = [kwargs['embed_size'], 64, 128, 256, 256]
        kwargs['kernel_sizes'] = [3, 3, 3, 3]
        kwargs['paddings'] = [1, 1, 1, 1]
        kwargs['relus'] = [True, True, True, True]
        kwargs['pools'] = [None, None, None, None]
        #######################################################

        self.Learner = BaseLearner(n, **kwargs)   # 基学习器内部含有beta
        self.LossFn = loss_fn
        self.MetaLr = lr

    def forward(self, support, query, sup_len, que_len, s_label):
        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(n*k, sup_seq_len)

        s_predict = self.Learner(support, sup_len)
        loss = self.LossFn(s_predict, s_label)
        self.Learner.zero_grad()        # 先清空基学习器梯度
        grads = t.autograd.grad(loss, self.Learner.parameters(), create_graph=True)
        adapted_state_dict = self.Learner.clone_state_dict()

        # 计算适应后的参数
        for (key, val), grad in zip(self.Learner.named_parameters(), grads):
            # 利用已有参数和每个参数对应的alpha调整系数来计算适应后的参数
            adapted_state_dict[key] = val - self.MetaLr * grad

        # 利用适应后的参数来生成测试集结果
        return self.Learner(query, que_len, params=adapted_state_dict).contiguous()

    def alpha(self, key):
        return self.Alpha[rename(key)]