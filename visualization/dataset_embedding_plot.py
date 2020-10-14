import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import torch as t
from torch.utils.data.dataloader import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,MDS,LocallyLinearEmbedding
import numpy as np

from models.FT import FT
from models.SIMPLE import SIMPLE
from components.datasets import SeqFileDataset
from utils.manager import PathManager, TrainingConfigManager
from utils.training import batchSequenceWithoutPad
from components.task import ImpEpisodeTask
from utils.magic import magicSeed

colors = ['darkgreen',
          'purple',
          'olive',
          'teal',
          'orangered',
          'yellowgreen',
          'steelblue',
          'gold',
          'magenta',
          'deepskyblue',
          'blueviolet',
          'red',
          'black',
          'bisque',
          'violet',
          'hotpink',
          'firebrick',
          'darkseagreen',
          'pink',
          'lime']

# ***********************************************************
dataset_name = "HKS"
dataset_subtype = 'test'
model_name = 'Random'
class_n = 20
version = 301
N = 20
figsize = (10,8)
axis_off = True
# ***************************************************************************

model_name_norm = 'FT' if model_name != 'SIMPLE' else model_name
path_man = PathManager(dataset=dataset_name, d_type=dataset_subtype,
                       model_name=model_name_norm,
                       version=version)
################################################
#----------------------读取模型参数------------------
################################################

if model_name == 'SIMPLE':
    model_cfg = TrainingConfigManager(path_man.Doc()+'config.json')
else:
    model_cfg = TrainingConfigManager('../run/runConfig.json')

modelParams = model_cfg.modelParams()

dataset = SeqFileDataset(path_man.FileData(), path_man.FileSeqLen(), N=N)
dataloader = DataLoader(dataset, batch_size=N, collate_fn=batchSequenceWithoutPad)

if model_name != 'Random':
    state_dict = t.load(path_man.Model() + '_v%s.0' % version)
    word_matrix = state_dict['Embedding.weight']
else:
    word_matrix = t.Tensor(np.load(path_man.WordEmbedMatrix(), allow_pickle=True))

loss_fn = t.nn.NLLLoss().cuda()

if model_name == 'SIMPLE':
    model = SIMPLE(word_matrix,
                   **modelParams)
    model.load_state_dict(state_dict)
elif model_name == 'FT':
    model = FT(class_n,
               loss_fn,
               word_matrix,
               **modelParams)
    model.load_state_dict(state_dict)
elif model_name == 'Random':
    model = FT(class_n,
               loss_fn,
               word_matrix,
               **modelParams)

model = model.cuda()
model.eval()

original_x = []

reduction = TSNE(n_components=2)
class_count = 0
for i, (x, _, lens) in enumerate(dataloader):
    class_count += 1
    x = x.cuda()
    x = model._embed(x, lens).cpu().view(N, -1).detach().numpy()
    original_x.append(x)

idxes = list(range(10))
original_x = np.array(original_x).reshape((class_count,N,-1))[idxes].reshape(len(idxes)*N,-1)

decom_x = reduction.fit_transform(original_x).reshape(len(idxes), N, 2)

plt.figure(figsize=figsize, dpi=300)
plt.axis('off')
for i in range(len(idxes)):
    plt.scatter(decom_x[i,:,0], decom_x[i,:,1],color=colors[i],marker='o',label=i)
plt.legend()
plt.show()

