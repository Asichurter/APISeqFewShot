import torch as t
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
import random as rd

from components.samplers import EpisodeSamlper
from utils.magic import magicSeed, magicList

def batchSequences(data):
    seqs = [x[0] for x in data]
    labels = t.LongTensor([x[1] for x in data])

    seqs.sort(key=lambda x: len(x), reverse=True)  # 按长度降序排列
    seq_len = [len(q) for q in seqs]
    seqs = pad_sequence(seqs, batch_first=True)

    return seqs, labels, seq_len

class EpisodeTask:
    def __init__(self, k, qk, n, N, dataset, cuda=True, label_expand=False):
        self.UseCuda = cuda
        self.Dataset = dataset
        self.Expand = label_expand

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

        rd.seed(task_seed)
        seed_for_each_class = rd.sample(magicList(), len(label_space))

        support_sampler = EpisodeSamlper(k, qk, N, seed_for_each_class,
                                         mode='support', label_space=label_space, shuffle=False)
        query_sampler = EpisodeSamlper(k, qk, N, seed_for_each_class,
                                       mode='query', label_space=label_space, shuffle=True)

        return support_sampler, query_sampler

    def getEpisodeData(self, support_sampler, query_sampler):
        k, qk, n, N = self.readParams()

        support_loader = DataLoader(self.Dataset, batch_size=k * n, sampler=support_sampler, collate_fn=batchSequences)
        query_loader = DataLoader(self.Dataset, batch_size=qk * n, sampler=query_sampler, collate_fn=batchSequences)

        supports, support_labels, support_lens = support_loader.__iter__().next()
        queries, query_labels, query_lens = query_loader.__iter__().next()

        # 将序列长度信息存储便于pack
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
            que_labels = t.argmax((sup_labels == que_labels).view(-1, n), dim=1)
            return que_labels.long()
        else:
            que_labels = (que_labels == sup_labels).view(-1, n)
            return que_labels.float()

    def episode(self, pack=True):
        label_space = self.getLabelSpace()
        support_sampler, query_sampler = self.getTaskSampler(label_space)
        supports, support_labels, queries, query_labels = self.getEpisodeData(support_sampler, query_sampler)

        if pack:
            supports = pack_padded_sequence(supports, self.SupSeqLenCache, batch_first=True)
            queries = pack_padded_sequence(queries, self.QueSeqLenCache, batch_first=True)

        labels = self.taskLabelNormalize(support_labels, query_labels)
        self.LabelsCache = labels

        if self.UseCuda:
            supports = supports.cuda()
            queries = queries.cuda()
            labels = labels.cuda()

        return supports, queries, labels

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