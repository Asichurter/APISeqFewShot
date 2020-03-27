import torch as t
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
import random as rd
import time

from components.samplers import EpisodeSamlper
from utils.magic import magicSeed, randomList
from utils.training import getBatchSequenceFunc

#########################################
# 基于Episode训练的任务类，包含采样标签空间，
# 采样实验样本，使用dataloader批次化序列样
# 本并且将任务的标签标准化。
#
# 调用episode进行任务采样和数据构建。
# 在采样时会自动缓存labels，输入模型的输出
# 调用accuracy计算得到正确率
#########################################
class EpisodeTask:
    def __init__(self, k, qk, n, N, dataset, cuda=True, label_expand=False, d_type='long'):
        self.UseCuda = cuda
        self.Dataset = dataset
        self.Expand = label_expand
        self.DType = d_type

        assert k + qk <= N, '支持集和查询集采样总数大于了类中样本总数!'
        self.Params = {'k': k, 'qk': qk, 'n': n, 'N': N}

        self.SupSeqLenCache = None
        self.QueSeqLenCache = None

        self.LabelsCache = None

    def readParams(self):
        params = self.Params
        k, qk, n, N = params['k'], params['qk'], params['n'], params['N']

        return k, qk, n, N

    def getLabelSpace(self):
        rd.seed(magicSeed())
        classes_list = [i for i in range(self.Dataset.ClassNum)]
        sampled_classes = rd.sample(classes_list, self.Params['n'])

        # print('label space: ', sampled_classes)
        return sampled_classes

    def getTaskSampler(self, label_space):
        task_seed = magicSeed()
        k, qk, n, N = self.readParams()

        # rd.seed(task_seed)
        # seed_for_each_class = rd.sample(magicList(), len(label_space))
        seed_for_each_class = randomList(num=len(label_space),
                                         seed=task_seed,
                                         allow_duplicate=True)  # rd.sample(magicList(), len(label_space))

        support_sampler = EpisodeSamlper(k, qk, N, seed_for_each_class,
                                         mode='support', label_space=label_space, shuffle=False)
        query_sampler = EpisodeSamlper(k, qk, N, seed_for_each_class,
                                       mode='query', label_space=label_space, shuffle=True)

        return support_sampler, query_sampler

    def getEpisodeData(self, support_sampler, query_sampler):
        k, qk, n, N = self.readParams()

        support_loader = DataLoader(self.Dataset, batch_size=k * n,
                                    sampler=support_sampler, collate_fn=getBatchSequenceFunc(d_type=self.DType))
        query_loader = DataLoader(self.Dataset, batch_size=qk * n,
                                  sampler=query_sampler, collate_fn=getBatchSequenceFunc(d_type=self.DType))

        supports, support_labels, support_lens = support_loader.__iter__().next()
        queries, query_labels, query_lens = query_loader.__iter__().next()

        # 将序列长度信息存储便于unpack
        self.SupSeqLenCache = support_lens
        self.QueSeqLenCache = query_lens

        return supports, support_labels, queries, query_labels

    def taskLabelNormalize(self, sup_labels, que_labels):
        k, qk, n, N = self.readParams()

        # 由于分类时是按照类下标与支持集进行分类的，因此先出现的就是第一类，每k个为一个类
        # size: [ql*n]
        sup_labels = sup_labels[::k].repeat(len(que_labels))  # 支持集重复q长度次，代表每个查询都与所有支持集类比较
        que_labels = que_labels.view(-1, 1).repeat((1, n)).view(-1)  # 查询集重复n次

        assert sup_labels.size(0) == que_labels.size(0), \
            '扩展后的支持集和查询集标签长度: (%d, %d) 不一致!' % (sup_labels.size(0), que_labels.size(0))

        # 如果进行扩展的话，每个查询样本的标签都会是n维的one-hot（用于交叉熵）
        # 不扩展是个1维的下标值（用于MSE）
        if not self.Expand:
            que_labels = t.argmax((sup_labels == que_labels).view(-1, n).int(), dim=1)
            return que_labels.long()
        else:
            que_labels = (que_labels == sup_labels).view(-1, n).int()
            return que_labels.float()

    def episode(self):
        raise NotImplementedError

    def labels(self):
        return self.LabelsCache

    def accuracy(self, out):
        k, qk, n, N = self.readParams()

        labels = self.LabelsCache

        if not self.Expand:
            out = t.argmax(out, dim=1)
        else:
            out = t.argmax(out.view(-1, n))
            labels = t.argmax(labels, dim=1)

        acc = (labels==out).sum().item() / labels.size(0)

        return acc


class ProtoEpisodeTask(EpisodeTask):
    def __init__(self, k, qk, n, N, dataset, cuda=True, label_expand=False):
        super(ProtoEpisodeTask, self).__init__(k, qk, n, N, dataset, cuda, label_expand)

    def episode(self):
        k, qk, n, N = self.readParams()

        label_space = self.getLabelSpace()
        support_sampler, query_sampler = self.getTaskSampler(label_space)
        supports, support_labels, queries, query_labels = self.getEpisodeData(support_sampler, query_sampler)

        # 已修正：因为支持集和查询集的序列长度因为pack而长度不一致，需要分开
        sup_seq_len = supports.size(1)
        que_seq_len = queries.size(1)

        labels = self.taskLabelNormalize(support_labels, query_labels)
        self.LabelsCache = labels

        if self.UseCuda:
            supports = supports.cuda()
            queries = queries.cuda()
            labels = labels.cuda()

        # 重整数据结构，便于模型读取任务参数
        supports = supports.view(n, k, sup_seq_len)
        queries = queries.view(n*qk, que_seq_len)      # 注意，此处的qk指每个类中的查询样本个数，并非查询集长度

        return (supports, queries, self.SupSeqLenCache, self.QueSeqLenCache), labels


class MatrixProtoEpisodeTask(EpisodeTask):
    def __init__(self, k, qk, n, N, dataset,
                 cuda=True,
                 label_expand=False,
                 unsqueeze=True):
        super(MatrixProtoEpisodeTask, self).__init__(k, qk, n, N, dataset, cuda, label_expand, d_type='float')
        self.Unsqueeze = unsqueeze

    def episode(self):
        k, qk, n, N = self.readParams()

        label_space = self.getLabelSpace()
        support_sampler, query_sampler = self.getTaskSampler(label_space)
        supports, support_labels, queries, query_labels = self.getEpisodeData(support_sampler, query_sampler)

        # support/query shape: [batch, seq,]
        image_width = supports.size(2)
        image_height = supports.size(3)
        sup_seq_len = supports.size(1)
        que_seq_len = queries.size(1)

        labels = self.taskLabelNormalize(support_labels, query_labels)
        self.LabelsCache = labels

        if self.UseCuda:
            supports = supports.cuda()
            queries = queries.cuda()
            labels = labels.cuda()

        # 重整数据结构，便于模型读取任务参数
        supports = supports.view(n, k, sup_seq_len, image_width, image_height)
        queries = queries.view(n*qk, que_seq_len, image_width, image_height)      # 注意，此处的qk指每个类中的查询样本个数，并非查询集长度

        if self.Unsqueeze:
            supports = supports.unsqueeze(2)
            queries = queries.unsqueeze(1)

        return (supports, queries, self.SupSeqLenCache, self.QueSeqLenCache), labels

class ImageProtoEpisodeTask(EpisodeTask):
    def __init__(self, k, qk, n, N, dataset, cuda=True, label_expand=False):
        super(ImageProtoEpisodeTask, self).__init__(k, qk, n, N, dataset, cuda, label_expand)

    def getEpisodeData(self, support_sampler, query_sampler):
        k, qk, n, N = self.readParams()

        support_loader = DataLoader(self.Dataset, batch_size=k * n, sampler=support_sampler)
        query_loader = DataLoader(self.Dataset, batch_size=qk * n, sampler=query_sampler)

        supports, support_labels = support_loader.__iter__().next()
        queries, query_labels = query_loader.__iter__().next()

        return supports, support_labels, queries, query_labels

    def episode(self):
        k, qk, n, N = self.readParams()

        label_space = self.getLabelSpace()
        support_sampler, query_sampler = self.getTaskSampler(label_space)
        supports, support_labels, queries, query_labels = self.getEpisodeData(support_sampler, query_sampler)

        # 已修正：因为支持集和查询集的序列长度因为pack而长度不一致，需要分开
        image_width = supports.size(2)
        image_height = supports.size(3)

        labels = self.taskLabelNormalize(support_labels, query_labels)
        self.LabelsCache = labels

        if self.UseCuda:
            supports = supports.cuda()
            queries = queries.cuda()
            labels = labels.cuda()

        # 重整数据结构，便于模型读取任务参数
        supports = supports.view(n, k, 1, image_width, image_height)
        queries = queries.view(n*qk, 1, image_width, image_height)      # 注意，此处的qk指每个类中的查询样本个数，并非查询集长度

        return (supports, queries), labels

def sampleLabelSpace(dataset, n):
    rd.seed(magicSeed())
    classes_list = [i for i in range(dataset.ClassNum)]
    sampled_classes = rd.sample(classes_list, n)

    return sampled_classes

def getTaskSampler(label_space, k, qk, N):
    task_seed = magicSeed()

    seed_for_each_class = randomList(num=len(label_space), seed=task_seed, allow_duplicate=True)#rd.sample(magicList(), len(label_space))

    support_sampler = EpisodeSamlper(k, qk, N, seed_for_each_class,
                                     mode='support', label_space=label_space, shuffle=False)
    query_sampler = EpisodeSamlper(k, qk, N, seed_for_each_class,
                                   mode='query', label_space=label_space, shuffle=True)

    return support_sampler, query_sampler

def getRNSampler(classes, train_num, test_num, num_per_class, seed=None):
    if seed is None:
        seed = time.time()%1000000

    assert train_num+test_num <= num_per_class, "单类中样本总数:%d少于训练数量加测试数量:%d！"%(num_per_class, train_num+test_num)

    # 先利用随机种子生成类中的随机种子
    rd.seed(seed)
    instance_seeds = rd.sample([i for i in range(100000)], len(classes))

    return EpisodeSamlper(train_num, test_num, num_per_class, instance_seeds, 'support', classes,  False),\
           EpisodeSamlper(train_num, test_num, num_per_class, instance_seeds, 'query', classes,  True)

# class A:
#     def __init__(self, x, y):
#         self.x = x,y
#
#     def print_x(self):
#         print(self.x)
#
# class AA(A):
#     def __init__(self, x, y):
#         super(AA, self).__init__(x, y)
#
# obj = AA(1,'a')
# obj.print_x()