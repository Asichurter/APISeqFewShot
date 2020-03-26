import torch as t
from torch.utils.data import Dataset
import numpy as np
import random as rd

from utils.file import loadJson

#########################################
# 基于已序列化的文件数据的数据集。会根据长度字典
# 将数据还原为原始长度来节省内存。
#########################################
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




#########################################
# 基于已序列化的文件数据的数据集。会根据长度字典
# 将数据还原为原始长度来节省内存。
#########################################
class MatrixSeqFileDataset(Dataset):

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





class ImageFileDataset(Dataset):
    # 直接指向support set或者query set路径下
    def __init__(self, base, N, rd_crop_size=None, rotate=True, squre=True):
        self.Data = np.load(base, allow_pickle=True)
        self.Label = []
        self.CropSize = rd_crop_size
        self.Rotate = rotate
        self.Width = self.Data.shape[2] if squre else None
        self.ClassNum = len(self.Data) // N

        for i in range(self.ClassNum):
            self.Label += [i]*N
        assert len(self.Label)==len(self.Data), "数据和标签长度不一致!(%d,%d)"%(len(self.Label),len(self.Data))

    def __getitem__(self, index):
        w = self.Width
        crop = self.CropSize
        img = t.FloatTensor(self.Data[index])
        if crop is not None:
            assert self.Width is not None and self.Data.shape[2]==self.Data.shape[3], "crop不能作用在非正方形图像上!"
            bound_width = w-crop
            x_rd,y_rd = rd.randint(0,bound_width),rd.randint(0,bound_width)
            img = img[:, x_rd:x_rd+crop, y_rd:y_rd+crop]
        # 依照论文代码中的实现，为了增加泛化能力，使用随机旋转
        if self.Rotate:
            rotation = rd.choice([0,1,2,3])
            img = t.rot90(img, k=rotation, dims=(1,2))
        label = self.Label[index]

        return img,label

if __name__ == '__main__':
    d = SeqFileDataset('D:/peimages/JSONs/virushare_20/data/train/data.npy',
                       'D:/peimages/JSONs/virushare_20/data/train/seqLength.json',
                       N=20)