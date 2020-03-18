from torch.utils.data import Sampler
import random as rd

from utils.magic import magicSeed

class EpisodeSamlper(Sampler):
    def __init__(self, k, qk, N, class_seeds, mode, label_space, shuffle=False):
        '''
        用于组成训练时sample set/query set和测试时support set和test set的采样器\n
        sample和query来自相同的类中，均为采样得到的\n
        :param label_space: 选到的要采样的类
        :param N: 每个类的最大样本数量
        :param shuffle: 是否随机打乱顺序
        '''
        self.LabelSpace = label_space
        self.N = N
        self.shuffle = shuffle
        self.instances = dict.fromkeys(label_space)

        if mode == 'support':
            # 为每一个类，根据其种子生成抽样样本的下标
            for cla,seed in zip(label_space, class_seeds):
                rd.seed(seed)
                # 注入该类对应的种子，然后抽样训练集
                self.instances[cla] = set(rd.sample([i for i in range(N)], k))

        elif mode == 'query':
            for cla,seed in zip(label_space, class_seeds):
                rd.seed(seed)
                # 注入与训练集同类的种子，保证训练集采样相同
                train_instances = set(rd.sample([i for i in range(N)], k))
                # 查询集与训练集不同
                test_instances = set([i for i in range(N)]).difference(train_instances)

                rd.seed(magicSeed())
                self.instances[cla] = rd.sample(test_instances, qk)

    def __iter__(self):
        batch = []
        for c,instances in self.instances.items():
            for i in instances:
                batch.append(self.N*c+i)
        if self.shuffle:
            rd.seed(magicSeed())
            rd.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1