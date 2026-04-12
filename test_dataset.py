from lib.train.base_functions import names2datasets
from lib.train.data import opencv_loader
from easydict import EasyDict

settings = EasyDict()
settings.use_lmdb = False
settings.env = EasyDict()

datasets = names2datasets(['COMPETITION_UAV'], settings, opencv_loader)
print('Number of sequences:', datasets[0].get_num_sequences())
print('Dataset name:', datasets[0].get_name())
seq_info = datasets[0].get_sequence_info(0)
print('First sequence frames:', seq_info['bbox'].shape[0])
print('Dataset loaded successfully!')