import os
import sys
import argparse
# Add HiT to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.config.HiT.config import cfg, update_config_from_file
from lib.train.base_functions import build_dataloaders, get_optimizer_scheduler, update_settings
from lib.models.HiT import build_hit
from lib.train.trainers import LTRTrainer
import lib.train.actors as actors
import torch
from easydict import EasyDict
from lib.utils.box_ops import giou_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune HiT-Base on competition data')
    parser.add_argument('--config', default='HiT_Base_finetune', help='config name')
    parser.add_argument('--save_dir', default='../output', help='save directory')
    parser.add_argument('--resume', action='store_true', help='resume training')
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = os.path.join('experiments', 'HiT', f'{args.config}.yaml')
    update_config_from_file(config_path)
    settings = EasyDict()
    settings.local_rank = -1
    settings.use_lmdb = False
    settings.env = EasyDict()
    settings.env.workspace_dir = r'F:\Cornea\HiT'
    settings.env.tensorboard_dir = r'F:\Cornea\output\tensorboard'
    settings.env.pretrained_networks = r'F:\Cornea\HiT\pretrained_networks'
    settings.save_dir = os.path.join(args.save_dir, 'checkpoints', 'train', 'HiT', args.config)
    settings.env.save_dir = settings.save_dir
    settings.use_gpu = True
    settings.project_path = 'HiT/HiT_Base_finetune'
    settings.project_path_full = os.path.join(settings.env.tensorboard_dir, 'HiT', 'HiT_Base_finetune')
    settings.log_file = r'F:\Cornea\output\training_log.txt'
    settings.script_name = 'HiT_Base_finetune'
    settings.description = 'Fine-tuning HiT-Base on competition UAV data'
    os.makedirs(settings.save_dir, exist_ok=True)
    os.makedirs(settings.env.tensorboard_dir, exist_ok=True)
    update_settings(settings, cfg)
    print('Building model...')
    model = build_hit(cfg)
    checkpoint_path = r'F:\Cornea\output\checkpoints\train\HiT\HiT_Base\HiT_Base.pth'
    print(f'Loading pretrained weights from {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['net'], strict=True)
    print('Pretrained weights loaded successfully.')
    model = model.cuda()
    print('Building dataloaders...')
    loader_train, loader_val = build_dataloaders(cfg, settings)
    optimizer, lr_scheduler = get_optimizer_scheduler(model, cfg)
    objective = {
        'giou': giou_loss,
        'l1': torch.nn.SmoothL1Loss()
    }
    loss_weight = {
        'giou': cfg.TRAIN.GIOU_WEIGHT,
        'l1': cfg.TRAIN.L1_WEIGHT
    }
    actor = actors.HiTActor(net=model, objective=objective, loss_weight=loss_weight, settings=settings)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
    start_epoch = 1
    if args.resume:
        latest = os.path.join(settings.save_dir, 'latest_checkpoint.pth')
        if os.path.exists(latest):
            print(f'Resuming from {latest}')
            trainer.load_checkpoint(latest)
            start_epoch = trainer.epoch + 1
    print(f'Starting fine-tuning for {cfg.TRAIN.EPOCH} epochs...')
    trainer.train(cfg.TRAIN.EPOCH, load_latest=args.resume, fail_safe=True)
    print('Fine-tuning complete!')

if __name__ == '__main__':
    main()