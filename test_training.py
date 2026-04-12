import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.config.HiT.config import cfg, update_config_from_file
from lib.train.base_functions import build_dataloaders, update_settings
from easydict import EasyDict

update_config_from_file('experiments/HiT/HiT_Base_finetune.yaml')

settings = EasyDict()
settings.local_rank = -1
settings.use_lmdb = False
settings.use_gpu = True
settings.project_path = 'HiT/HiT_Base_finetune'
settings.env = EasyDict()
settings.env.workspace_dir = r'F:\Cornea\HiT'
settings.env.tensorboard_dir = r'F:\Cornea\output\tensorboard'
settings.env.pretrained_networks = r'F:\Cornea\HiT\pretrained_networks'
settings.save_dir = r'F:\Cornea\output\checkpoints\train\HiT\HiT_Base_finetune'
settings.env.save_dir = settings.save_dir
os.makedirs(settings.save_dir, exist_ok=True)
os.makedirs(settings.env.tensorboard_dir, exist_ok=True)
update_settings(settings, cfg)

print('Building dataloaders...')
loader_train, loader_val = build_dataloaders(cfg, settings)

print('Testing first batch...')
for i, batch in enumerate(loader_train):
    print(f'Batch {i} loaded successfully!')
    print(f'Template shape: {batch["template_images"].shape}')
    print(f'Search shape: {batch["search_images"].shape}')
    if i >= 2:
        break

print('Data pipeline works!')