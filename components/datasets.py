import torch as t
from torch.utils.data import Dataset

from utils.file import loadJson


class SeqFileDataset(Dataset):

    def __init__(self, data_path, seq_path, N):
        self.Data = t.load(data_path)
        self.SeqLength = loadJson(seq_path)
        self.Label = []
        self.ClassNum = len(self.Data) // N

        assert len(self.Data) % N == 0, \
            '数据总长度%d不是指定每个类样本数量%d的整倍数！' % (len(self.Data), N)

        assert len(self.Data) == len(self.SeqLength), \
            '数据总长度%d与序列长度数据总长度%d不同' % (len(self.Data), len(self.SeqLength))

        for i in range(len(self.Data) // N):
            self.Label += [i] * N

        data_list = [tensor for tensor in self.Data]    # 使用列表装非等长张量
        for i, length in self.SeqLength.items():
            idx = int(i)
            data_list[idx] = data_list[idx][:length]        # 根据序列长度文件中的数据，只取pad前的数据

        self.Data = data_list

    def __getitem__(self, item):
        # print(item)
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Data)






if __name__ == '__main__':
    d = SeqFileDataset('D:/peimages/JSONs/virushare_20/data/train/data.npy',
                       'D:/peimages/JSONs/virushare_20/data/train/seqLength.json',
                       N=20)