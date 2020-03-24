from scripts.dataset import makeDataFile
from utils.manager import PathManager

manager = PathManager(dataset='virushare_20', d_type='test')
makeDataFile(json_path=manager.Folder(),
             w2idx_path=manager.WordIndexMap(),
             seq_length_save_path=manager.FileSeqLen(),
             data_save_path=manager.FileData(),
             num_per_class=20,
             max_seq_len=100)