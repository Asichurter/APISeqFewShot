import torch as t
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn
from torch.optim.adam import Adam

from components.reduction.selfatt import BiliAttnReduction
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder
from components.sequence.TCN import TemporalConvNet
from utils.training import extractTaskStructFromInput, getMaskFromLens, \
                            collectParamsFromStateDict

def rename(name, token='-'):
    '''
    由于ParameterDict中不允许有.字符的存在，因此利用本方法来中转，将
    .字符转化为-来避开
    '''
    return name.replace('.',token)

class BaseLearner(nn.Module):
    def __init__(self, n, pretrained_matrix, embed_size, seq_len, **modelParams):
        super(BaseLearner, self).__init__()
        # 需要adapt的参数名称
        self.adapted_keys = [
                            # 'Attention.IntAtt.weight',
                            # 'Attention.ExtAtt.weight',
                            'Attention.weight',
                            'fc.weight',
                            'fc.bias']
        self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix,
                                                      padding_idx=0)
        # self.EmbedNorm = nn.LayerNorm(embed_size)
        self.EmbedDrop = nn.Dropout(modelParams['dropout'])

        self.Encoder = BiLstmEncoder(input_size=embed_size, **modelParams)#CNNEncoder1D(**kwargs)
        # self.Encoder = TemporalConvNet(**kwargs)
        directions = 1 + modelParams['bidirectional']

        self.Attention = nn.Linear(directions*modelParams['hidden_size'], 1, bias=False)
        # self.Attention = nn.Linear(2*kwargs['hidden_size'], 1, bias=False)
        # self.Attention = AttnReduction(input_dim=2*kwargs['hidden_size'])

        # out_size = kwargs['hidden_size']
        # self.fc = nn.Linear(seq_len, n)
        self.fc = nn.Linear(directions*modelParams['hidden_size'], n)
                                                    # 对于双向lstm，输出维度是隐藏层的两倍
                                                    # 对于CNN，输出维度是嵌入维度

    def forward(self, x, lens, params=None):
        length = x.size(0)
        x = self.Embedding(x)
        x = self.EmbedDrop(x)
        # x = self.EmbedNorm(x)
        x = self.Encoder(x, lens)

        # shape: [batch, seq, dim] => [batch, mem_step, dim]
        dim = x.size(2)
        mem_step_len = x.size(1)

        if params is None:
            att_weight = self.Attention(x).repeat((1,1,dim))
            len_expansion = lens.unsqueeze(1).repeat((1,dim)).cuda()
            # c = ∑ αi·si / T = ∑(θ·si)·si / T
            x = (x * att_weight).sum(dim=1) / len_expansion
            # x = self.Attention(x)

            x = x.view(length, -1)
            x = self.fc(x)
        else:
            # 在ATAML中，只有注意力权重和分类器权重是需要adapt的对象
            att_weight = F.linear(x, weight=params['Attention.weight']).repeat((1,1,dim))
            len_expansion = lens.unsqueeze(1).repeat((1,dim)).cuda()
            # # c = ∑ αi·si / T = ∑(θ·si)·si / T
            x = (x * att_weight).sum(dim=1) / len_expansion

            # x = self.Attention.static_forward(x,
            #                                   params=[params['Attention.IntAtt.weight'],
            #                                           params['Attention.ExtAtt.weight']],
            #                                   lens=lens)

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
            for key, val in self.named_parameters()
            if key in self.adapted_keys
        }
        return cloned_state_dict

    ##############################################
    # 获取本模型中需要被adapt的参数，named参数用于指定是否
    # 在返回时附带参数名称
    ##############################################
    def adapt_parameters(self, with_named=False):
        parameters = []
        for n,p in self.named_parameters():
            if n in self.adapted_keys:
                if with_named:
                    parameters.append((n,p))
                else:
                    parameters.append(p)
        return parameters

class ATAML(nn.Module):
    def __init__(self, n, loss_fn, inner_lr=3e-2, method='maml',
                 adapt_iter=2, **modelParams):
        super(ATAML, self).__init__()

        self.DataParallel = modelParams['data_parallel'] if 'data_parallel' in modelParams else False

        #######################################################
        # For CNN only
        # kwargs['dims'] = [kwargs['embed_size'], 64, 128, 256, 256]
        # kwargs['kernel_sizes'] = [3, 3, 3, 3]
        # kwargs['paddings'] = [1, 1, 1, 1]
        # kwargs['relus'] = [True, True, True, True]
        # kwargs['pools'] = [None, None, None, None]
        #######################################################

        self.Learner = BaseLearner(n, seq_len=50, **modelParams)   # 基学习器内部含有beta
        self.LossFn = loss_fn
        self.MetaLr = inner_lr
        self.AdaptedPar = None
        self.Method = method
        self.AdaptIter = adapt_iter

    def forward(self, support, query, sup_len, que_len, support_labels,
                adapt_iter=1):
        method = self.Method

        if self.DataParallel:
            support = support.squeeze(0)
            sup_len = sup_len[0]
            support_labels = support_labels[0]

        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(n*k, sup_seq_len)

        # ---------------------------------------------------------------
        # fix the bug, which: reset the 'adapted_par' in every adapt iteration
        # ---------------------------------------------------------------
        adapted_state_dict = self.Learner.clone_state_dict()
        for n,p in adapted_state_dict.items():
            p.requires_grad_(True)

        for a_i in range(self.AdaptIter):
            # ---------------------------------------------------------------
            # fix the bug, which: use original parameters instead of the adapted
            # ones in every adapt iteration
            # ---------------------------------------------------------------
            adapted_pars = collectParamsFromStateDict(adapted_state_dict)

            s_predict = self.Learner(support, sup_len, params=adapted_state_dict)
            loss = self.LossFn(s_predict, support_labels)
            grads = t.autograd.grad(loss, adapted_pars, create_graph=True)

            # 计算适应后的参数
            for (key, val), grad in zip(adapted_state_dict.items(), grads):
                adapted_state_dict[key] = val - self.MetaLr * grad

        if method == 'fomaml':
            adapted_par = collectParamsFromStateDict(adapted_state_dict)

            # 对于一阶MAML，需要查询集loss和适应后参数来计算梯度
            return self.Learner(query, que_len, params=adapted_state_dict), adapted_par

        # 对于vanilia MAML，只需要query loss
        else:
            return self.Learner(query, que_len, params=adapted_state_dict)


    def alpha(self, key):
        return self.Alpha[rename(key)]