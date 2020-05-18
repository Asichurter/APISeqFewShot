import torch as t

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn
from torch.optim.adam import Adam

from components.sequence.CNN import CNNBlock1D


def rename(name, token='-'):
    '''
    由于ParameterDict中不允许有.字符的存在，因此利用本方法来中转，将
    .字符转化为-来避开
    '''
    return name.replace('.',token)

class BaseLearner(nn.Module):
    def __init__(self, n, **kwargs):
        super(BaseLearner, self).__init__()
        weight = t.randn((10,64))
        self.Embedding = nn.Embedding.from_pretrained(weight,
                                                      freeze=False,
                                                      padding_idx=0)
        # self.EmbedNorm = nn.LayerNorm(64)

        self.channels = [64, 64, 128, 256, 256]
        self.strides = [1, 1, 1, 1]
        self.kernels = [3, 3, 3, 3]
        self.paddings = [1, 1, 1, 1]
        self.pools = ['max', 'max', 'max', 'ada']
        layers = [CNNBlock1D(
            in_feature=self.channels[i],
            out_feature=self.channels[i+1],
            stride=self.strides[i],
            kernel=self.kernels[i],
            padding=self.paddings[i],
            pool=self.pools[i]) for i in range(4)]
        self.Encoder = nn.Sequential(*layers)

        self.fc = nn.Linear(self.channels[-1], n)

    def forward(self, x, params=None):
        length = x.size(0)


        if params is None:
            x = self.Embedding(x)
            # x = self.EmbedNorm(x)
            x = x.transpose(1, 2).contiguous()
            x = self.Encoder(x).view(length, -1)
            x = self.fc(x)
        else:
            # 使用适应后的参数来前馈
            x = self.Embedding(x)
            # x = F.embedding(x, params['Embedding.weight'],padding_idx=0)
            # x = F.layer_norm(x, (64,), params['EmbedNorm.weight'], params['EmbedNorm.bias'])
            x = x.transpose(1, 2).contiguous()
            for i in range(4):      # conv-4结构
                x = F.conv1d(
                    x,
                    params['Encoder.%d.0.weight'%i],
                    stride=self.strides[i],
                    padding=self.paddings[i]
                )
                x = F.batch_norm(
                    x,
                    params['Encoder.%d.1.running_mean'%i],
                    params['Encoder.%d.1.running_var'%i],
                    params['Encoder.%d.1.weight'%i],
                    params['Encoder.%d.1.bias'%i],
                    momentum=1,
                    training=True)
                x = F.relu(x, inplace=True)

                if self.pools[i] == 'max':
                    x = F.max_pool1d(x, 2)
                else:
                    x = F.adaptive_max_pool1d(x, 1)

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

class MetaSGD(nn.Module):
    def __init__(self, n, loss_fn, lr=1e-3, **kwargs):
        super(MetaSGD, self).__init__()
        self.Learner = BaseLearner(n, **kwargs)   # 基学习器内部含有beta
        self.Alpha = nn.ParameterDict({
            rename(name):nn.Parameter(lr * t.ones_like(val, requires_grad=True))
            for name,val in self.Learner.named_parameters() if val.requires_grad
        })      # 初始化alpha
        self.LossFn = loss_fn

    def forward(self, support, query, s_label):
        # for support, query, s_label, q_label in zip(support_list, query_list, s_label_list, q_label_list):
        s_predict = self.Learner(support)
        loss = self.LossFn(s_predict, s_label)
        self.Learner.zero_grad()        # 先清空基学习器梯度
        grads = t.autograd.grad(loss, self.Learner.parameters(), create_graph=True)
        adapted_state_dict = self.Learner.clone_state_dict()

        # 计算适应后的参数
        for (key, val), grad in zip(self.Learner.named_parameters(), grads):
            # 利用已有参数和每个参数对应的alpha调整系数来计算适应后的参数
            adapted_state_dict[key] = val - self.alpha(key) * grad

        # 利用适应后的参数来生成测试集结果
        return self.Learner(query, params=adapted_state_dict)

    def alpha(self, key):
        return self.Alpha[rename(key)]

if __name__ == '__main__':
    loss_fn = nn.NLLLoss().cuda()
    model = MetaSGD(input_size=32, n=5, loss_fn=loss_fn).cuda()
    opt = Adam(model.parameters(), lr=1e-3)

    batch_size = 1
    supports = []
    queries = []
    support_labels = []
    query_labels = []

    for i in range(batch_size):
        supports.append(t.randint(low=1,high=10,size=(25,50),dtype=t.long, device='cuda:0'))
        support_labels.append(t.zeros((25,), dtype=t.long, device='cuda:0'))
        queries.append(t.randint(low=1,high=10,size=(75,50),dtype=t.long, device='cuda:0'))
        query_labels.append(t.ones((75,), dtype=t.long, device='cuda:0'))

    model.zero_grad()
    meta_loss = 0.
    epoch = 0
    for support, query, s_label, q_label in zip(supports, queries, support_labels, query_labels):
        print(epoch)
        epoch += 1

        predict = model(support, query, s_label)
        meta_loss += loss_fn(predict, q_label)

    opt.zero_grad()
    meta_loss.backward()
    opt.step()
    print('done！')