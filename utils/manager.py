


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
