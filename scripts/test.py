from components.task import EpisodeTask
from components.datasets import SeqFileDataset

k = 5
qk = 15
n = 5
N = 20

data_path = 'D:/peimages/JSONs/virushare_20/data/train/data.npy'
seq_path = 'D:/peimages/JSONs/virushare_20/data/train/seqLength.json'

dataset = SeqFileDataset(data_path, seq_path, N)
task = EpisodeTask(k, qk, n, N, dataset)

e = task.episode()