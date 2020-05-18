import torch as t
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn
from torch.optim.adam import Adam

from components.reduction.selfatt import AttnReduction
from components.sequence.CNN import CNNEncoder1D
from components.sequence.LSTM import BiLstmEncoder
from utils.training import extractTaskStructFromInput, getMaskFromLens, splitMetaBatch

def rename(name, token='-'):
    '''
    由于ParameterDict中不允许有.字符的存在，因此利用本方法来中转，将
    .字符转化为-来避开
    '''
    return name.replace('.',token)

class BaseLearner(nn.Module):
    def __init__(self, n, pretrained_matrix, embed_size, seq_len, **kwargs):
        super(BaseLearner, self).__init__()
        # 需要adapt的参数名称
        self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix,
                                                      freeze=False,
                                                      padding_idx=0)
        self.EmbedNorm = nn.LayerNorm(embed_size)
        self.Encoder = BiLstmEncoder(input_size=embed_size, **kwargs)#CNNEncoder1D(**kwargs)
        self.Attention = nn.Linear(2*kwargs['hidden_size'], 1, bias=False)
        # self.Attention = AttnReduction(input_dim=2*kwargs['hidden_size'])

        # out_size = kwargs['hidden_size']
        # self.fc = nn.Linear(seq_len, n)
        self.fc = nn.Linear(kwargs['hidden_size']*2, n)
                                                    # 对于双向lstm，输出维度是隐藏层的两倍
                                                    # 对于CNN，输出维度是嵌入维度

        self.adapted_keys = []
                            # [
                            # # 'Attention.IntAtt.weight',
                            # # 'Attention.ExtAtt.weight',
                            # 'Attention.weight',
                            # 'fc.weight',
                            # 'fc.bias']

        self.addAdaptedKeys()


    def addAdaptedKeys(self):
        for n,p in self.named_parameters():
            self.adapted_keys.append(n)

    def forward(self, x, lens, params=None):
        length = x.size(0)
        x = self.Embedding(x)
        x = self.EmbedNorm(x)
        x = self.Encoder(x, lens)

        # shape: [batch, seq, dim] => [batch, mem_step, dim]
        dim = x.size(2)
        mem_step_len = x.size(1)

        if params is None:
            att_weight = self.Attention(x).repeat((1,1,dim))
            len_expansion = t.Tensor(lens).unsqueeze(1).repeat((1,dim)).cuda()
            # c = ∑ αi·si / T = ∑(θ·si)·si / T
            x = (x * att_weight).sum(dim=1) / len_expansion
            # x = self.Attention(x)

            x = x.view(length, -1)
            x = self.fc(x)
        else:
            # 在ATAML中，只有注意力权重和分类器权重是需要adapt的对象
            att_weight = F.linear(x, weight=params['Attention.weight']).repeat((1,1,dim))
            len_expansion = t.Tensor(lens).unsqueeze(1).repeat((1,dim)).cuda()
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
            for key, val in self.state_dict().items()
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

    # def eval(self):
    #     self.Embedding.eval()
    #     self.EmbedNorm.eval()
    #     self.Encoder.train()
    #     self.Attention.eval()

class Reptile(nn.Module):
    def __init__(self, n, loss_fn, meta_lr=1, adapt_lr=5e-2, **kwargs):
        super(Reptile, self).__init__()

        #######################################################
        # For CNN only
        # kwargs['dims'] = [kwargs['embed_size'], 64, 128, 256, 256]
        # kwargs['kernel_sizes'] = [3, 3, 3, 3]
        # kwargs['paddings'] = [1, 1, 1, 1]
        # kwargs['relus'] = [True, True, True, True]
        # kwargs['pools'] = [None, None, None, None]
        #######################################################

        self.Learner = BaseLearner(n, seq_len=50, **kwargs)   # 基学习器内部含有beta
        self.LossFn = loss_fn
        self.MetaLr = meta_lr
        self.AdaLr = adapt_lr
        self.AdaptedPar = None

    def _train(self, n, k, max_sample_num, support, sup_len, s_label):
        pass

    def forward(self, n, k, support, query, sup_len, que_len, s_label):
        '''
            In Reptile, update can be done within each forward loop without
            evolvement of optimizer, so when calling this method with query
            provided, the parameters can be updated automatically.
        '''
        # n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        sup_seq_len = support.size(1)

        # 提取了任务结构后，将所有样本展平为一个批次
        support = support.view(-1, sup_seq_len)

        accum_grads = [t.zeros_like(p).cuda() for p in self.Learner.parameters()]

        # iterate through meta-mini-batches
        for sup_data_batch, sup_label_batch, sup_len_batch in splitMetaBatch(support, s_label,
                                                                             batch_num=5,
                                                                             max_sample_num=len(support)//n,
                                                                             sample_num=k,
                                                                             meta_len=sup_len):
            s_predict = self.Learner(sup_data_batch, sup_len_batch)
            loss = self.LossFn(s_predict, sup_label_batch)

            grads = t.autograd.grad(loss, self.Learner.parameters())

            # accumulate inner-loop step gradient
            for i,g in enumerate(grads):
                accum_grads[i] += self.AdaLr * g

        adapted_pars = self.Learner.clone_state_dict()

        # perform meta-update in Reptile
        for (key, ada_par), grad in zip(self.Learner.named_parameters(), accum_grads):
            adapted_pars[key] = adapted_pars[key] + self.MetaLr*(grad - adapted_pars[key])

        if query is not None:
            # 避免了LSTM的Functional调用
            origin_pars = self.Learner.clone_state_dict()
            self.Learner.load_state_dict(adapted_pars)
            res = self.Learner(query, que_len)              # , params=adapted_pars,
            self.Learner.load_state_dict(origin_pars)
            return res
        else:
            acc = (t.argmax(s_predict, dim=1)==sup_label_batch).sum().item() / s_predict.size(0)
            loss_val = loss.detach().item()
            self.Learner.load_state_dict(adapted_pars)      # no query indicates updating parameters in training stage

            return acc, loss

    # def eval(self):
    #     self.Learner.eval()