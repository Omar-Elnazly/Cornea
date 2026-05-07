import os
import torch
import importlib

# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss

# train pipeline related
from lib.train.trainers import LTRTrainer

# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP

# base helpers
from .base_functions import *

# network
from lib.models.ortrack import build_ortrack

# actor
from lib.train.actors import ORTrackActor

# loss
from ..utils.focal_loss import FocalLoss


# ======================================================
# ✅ PARTIAL FREEZE FUNCTION (ONLY DEFINED ONCE)
# ======================================================
def apply_partial_freeze(model, freeze_up_to=6):
    # Reset all backbone params to trainable
    for p in model.backbone.parameters():
        p.requires_grad_(True)

    # Freeze patch embedding
    for p in model.backbone.patch_embed.parameters():
        p.requires_grad_(False)

    # Freeze blocks 0 → freeze_up_to-1
    for i, block in enumerate(model.backbone.blocks):
        if i < freeze_up_to:
            for p in block.parameters():
                p.requires_grad_(False)


def run(settings):

    settings.description = 'Training script for ORTrack'

    # ======================================================
    # Load config
    # ======================================================
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)

    config_module = importlib.import_module(
        "lib.config.%s.config" % settings.script_name
    )

    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)

    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print(f"{key} configuration:", cfg[key])
            print()

    update_settings(settings, cfg)

    # ======================================================
    # Logging
    # ======================================================
    log_dir = os.path.join(settings.save_dir, 'logs')

    if settings.local_rank in [-1, 0]:
        os.makedirs(log_dir, exist_ok=True)

    settings.log_file = os.path.join(
        log_dir,
        f"{settings.script_name}-{settings.config_name}.log"
    )

    # ======================================================
    # Dataloaders
    # ======================================================
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if (
        "RepVGG" in cfg.MODEL.BACKBONE.TYPE
        or "swin" in cfg.MODEL.BACKBONE.TYPE
        or "LightTrack" in cfg.MODEL.BACKBONE.TYPE
    ):
        cfg.ckpt_dir = settings.save_dir

    # ======================================================
    # CREATE NETWORK
    # ======================================================
    if settings.script_name != "ortrack":
        raise ValueError("illegal script name")

    is_distill_training = cfg.MODEL.get('IS_DISTILL', False)

    # ---- student model ----
    net = build_ortrack(cfg)

    FREEZE_UP_TO_BLOCK = 6   # 👈 ADD THIS LINE (for checker)
    # ✅ APPLY FREEZE (ONLY ONCE)
    apply_partial_freeze(net, freeze_up_to=6)

    # ---- verification print ----
    frozen    = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total     = frozen + trainable

    print(f"Partial freeze applied: blocks 0-5 FROZEN, 6-11 TRAINABLE")
    print(f"Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.1f}%)")
    print(f"Frozen:    {frozen:,} / {total:,} params ({100 * frozen / total:.1f}%)")

    for i, block in enumerate(net.backbone.blocks):
        status = "TRAINABLE" if any(p.requires_grad for p in block.parameters()) else "FROZEN"
        print(f"  backbone.blocks.{i}: {status}")

    # ======================================================
    # TEACHER MODEL (optional)
    # ======================================================
    net_teacher = None

    if is_distill_training:

        cfg_teacher = cfg
        student_backbone = cfg_teacher.MODEL['BACKBONE']['TYPE']
        teacher_backbone = student_backbone.replace('_distilled', '')

        cfg_teacher.MODEL['BACKBONE']['TYPE'] = teacher_backbone
        net_teacher = build_ortrack(cfg_teacher)

        root_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../..')
        )

        checkpoint_path = os.path.join(
            root_path,
            f'teacher_model/{teacher_backbone}/ORTrack_ep0300.pth.tar'
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        net_teacher.load_state_dict(checkpoint["net"], strict=False)
        net_teacher.cuda()

        print(f"Loaded teacher checkpoint: {checkpoint_path}")

    # ======================================================
    # Device + DDP
    # ======================================================
    net.cuda()

    if settings.local_rank != -1:

        if net_teacher is not None:
            net_teacher = DDP(
                net_teacher,
                device_ids=[settings.local_rank],
                find_unused_parameters=True
            )

        net = DDP(
            net,
            device_ids=[settings.local_rank],
            find_unused_parameters=True
        )

        settings.device = torch.device(f"cuda:{settings.local_rank}")

    else:
        settings.device = torch.device("cuda:0")

    # ======================================================
    # Actor + Loss
    # ======================================================
    focal_loss = FocalLoss()

    objective = {
        'giou': giou_loss,
        'l1': l1_loss,
        'focal': focal_loss,
        'cls': BCEWithLogitsLoss()
    }

    loss_weight = {
        'giou': cfg.TRAIN.GIOU_WEIGHT,
        'l1': cfg.TRAIN.L1_WEIGHT,
        'focal': 1.,
        'cls': 1.0,
        'sim_loss': 0.05,
        'distill_loss': 0.00002
    }

    actor = ORTrackActor(
        net=net,
        objective=objective,
        loss_weight=loss_weight,
        settings=settings,
        cfg=cfg
    )

    if net_teacher is not None:
        actor.net_teacher = net_teacher

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    # ======================================================
    # Optimizer (ONLY trainable params)
    # ======================================================
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    use_amp = getattr(cfg.TRAIN, "AMP", False)

    trainer = LTRTrainer(
        actor,
        [loader_train, loader_val],
        optimizer,
        settings,
        lr_scheduler,
        use_amp=use_amp
    )

    trainer.actor.net.is_distill_training = is_distill_training

    # ======================================================
    # TRAIN
    # ======================================================
    trainer.train(
        cfg.TRAIN.EPOCH,
        load_latest=False,
        load_previous_ckpt=True,
        fail_safe=True
    )