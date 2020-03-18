import numpy as np
from time import time


##########################################################
# 项目数据集路径管理器。
# 可以使用方法获取数据集路径。
# d_type用于指定当前使用的是训练集，验证集还是测试集，指定为all时代表
# 三者均不是。
##########################################################
class PathManager:

    # 文件父目录
    ParentPath = 'D:/peimages/JSONs/'

    # 数据文件夹路径
    FolderPathTemp = '%s/%s/'

    # 词嵌入矩阵路径
    WordEmbedMatrixPathTemp = '%s/data/matrix.npy'

    # 词转下标表路径
    WordIndexMapPathTemp = '%s/data/wordMap.json'

    # 文件型数据路径
    FileDataPathTemp = '%s/data/%s/data.npy'
    # 文件数据对应的长度表路径
    FileSeqLenPathTemp = '%s/data/%s/seqLength.json'

    def __init__(self, dataset, d_type='all'):
        # 不允许在指定all时访问数据分割路径
        BanedListWhenAll = ['FileDataPath', 'FileSeqLenPath']

        self.DataType = d_type

        self.FolderPath = self.ParentPath + self.FolderPathTemp % (dataset, d_type)

        self.WordEmbedMatrixPath = self.ParentPath + self.WordEmbedMatrixPathTemp % dataset
        self.WordIndexMapPath = self.ParentPath + self.WordIndexMapPathTemp % dataset

        self.FileDataPath = self.ParentPath + self.FileDataPathTemp % (dataset, d_type)
        self.FileSeqLenPath = self.ParentPath + self.FileSeqLenPathTemp % (dataset, d_type)

    def Folder(self):
        return self.FolderPath

    def WordEmbedMatrix(self):
        return self.WordEmbedMatrixPath

    def WordIndexMap(self):
        return self.WordIndexMapPath

    def FileData(self):
        if self.DataType == 'all':
            raise ValueError('在数据分割为all时，不允许访问FileData')
        else:
            return self.FileDataPath

    def FileSeqLen(self):
        if self.DataType == 'all':
            raise ValueError('在数据分割为all时，不允许访问FileSeqLen')
        else:
            return self.FileSeqLenPath


#########################################
# 训练时的统计数据管理器。主要用于记录训练和验证的
# 正确率和损失值，并且根据选择的标准来选择最优模型
# 保存和打印训练数据。在记录验证数据时会打印训练和
# 验证信息
#########################################
class TrainStatManager:

    def __init__(self, train_report_iter=100, criteria='loss'):
        self.TrainHist = {'accuracy': [],
                          'loss': []}
        self.ValHist = {'accuracy': [],
                          'loss': []}

        assert criteria in self.TrainHist.keys(), '给定的标准 %s 不在支持范围内!'%criteria

        self.Criteria = criteria
        self.BestVal = float('inf') if criteria=='loss' else -1.
        self.BestValEpoch = -1
        self.TrainReportIter = train_report_iter
        self.TrainIterCount = -1

        self.TimeStamp = None

    def start(self):
        self.TimeStamp = time()

    def recordTraining(self, acc, loss):
        self.TrainHist['accuracy'].append(acc)
        self.TrainHist['loss'].append(loss)
        self.TrainIterCount += 1

    def recordValidating(self,acc, loss):
        self.ValHist['accuracy'].append(acc)
        self.ValHist['loss'].append(loss)

        if self.Criteria == 'loss':
            if loss < self.BestVal:
                self.BestValEpoch = self.TrainIterCount
                self.BestVal = loss
                # TODO 保存模型
        else:
            if acc > self.BestVal:
                self.BestValEpoch = self.TrainIterCount
                self.BestVal = acc
                # TODO 保存模型

        self.printRecentRecord()


    def printRecentRecord(self):
        average_train_acc = np.mean(self.TrainHist['accuracy'][-1*self.TrainReportIter:])
        average_train_loss = np.mean(self.TrainHist['loss'][-1*self.TrainReportIter:])
        recent_val_acc = self.ValHist['accuracy'][-1]
        recent_val_loss = self.ValHist['loss'][-1]

        new_time_stamp = time()

        print('***********************************')
        print('train acc: ', average_train_acc)
        print('train loss: ', average_train_loss)
        print('----------------------------------')
        print('val acc:', recent_val_acc)
        print('val loss:', recent_val_loss)
        print('----------------------------------')
        print('best val %s:'%self.Criteria, self.BestVal)
        print('best epoch:', self.BestValEpoch)
        print('time consuming:', new_time_stamp-self.TimeStamp)
        print('\n***********************************')
        print('\n\n%d -> %d epoches...'%(self.TrainIterCount, self.TrainIterCount+self.TrainReportIter))

        self.TimeStamp = new_time_stamp



