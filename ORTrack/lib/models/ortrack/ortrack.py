"""
Basic ORTrack model -> lib/models/ortrack/ortrack.py.
"""
import os
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh

import numpy as np
from scipy.stats import multivariate_normal
import cv2

from lib.models.ortrack.deit import (
    deit_tiny_patch16_224,
    deit_tiny_patch16_224_distill,
)
from lib.models.ortrack.vision_transformer import (
    vit_tiny_patch16_224,
    vit_tiny_distilled_patch16_224,
)
from lib.models.ortrack.eva import (
    eva02_tiny_patch14_224,
    eva02_tiny_patch14_224_distill,
)


class ORTrack(nn.Module):
    """Base ORTrack model"""

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        super().__init__()

        self.backbone = transformer
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type

        if head_type in ["CORNER", "CENTER"]:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_sz_t = int(box_head.feat_template_sz)
            self.feat_len_s = self.feat_sz_s ** 2
            self.feat_len_t = self.feat_sz_t ** 2

        if aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.intensity = []
        self.randomMask = False

    # =========================================================
    # ---------------- MASKING FUNCTIONS ----------------------
    # =========================================================

    def random_masking(self, N, H, W, D, mask_ratio, device):

        len_keep = int(H * W * (1 - mask_ratio))

        noise = torch.rand(N, H * W, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([N, H * W], device=device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.view(N, H, W)

        return mask

    def masking(self, template, block_sz, mask_ratio, device):

        N, D, H, W = template.shape

        assert H % block_sz == 0 and W % block_sz == 0

        h = H // block_sz
        w = W // block_sz

        mask = self.random_masking(N, h, w, D, mask_ratio, device)

        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(1),
            size=(H, W),
            mode="nearest",
        )

        return mask

    def masking_CoxProcess_pytorch(self, template, mask_ratio=0.5, block_size=16):
        """
        Returns a binary spatial mask [B, 1, H, W].
        The caller is responsible for multiplying into the template.
        This avoids the double-channel-expand bug (tensor 3 vs 9).
        """
        B, C, H, W = template.shape
        device = template.device

        num_blocks_h = H // block_size
        num_blocks_w = W // block_size
        num_blocks   = num_blocks_h * num_blocks_w
        K = max(1, int(round((1.0 - mask_ratio) * num_blocks)))

        # Gaussian intensity centred on patch grid (float32 — avoids .mean() Long error)
        gy, gx = torch.meshgrid(
            torch.arange(num_blocks_h, device=device, dtype=torch.float32),
            torch.arange(num_blocks_w, device=device, dtype=torch.float32),
            indexing='ij',
        )
        intensity = torch.exp(
            -((gy - gy.mean()) ** 2 + (gx - gx.mean()) ** 2)
        ).reshape(-1)

        masks = []
        for _ in range(B):
            poisson_intensity = torch.poisson(intensity * K) + 1e-6
            keep = torch.multinomial(
                poisson_intensity,
                num_samples=min(K, num_blocks),
                replacement=False,
            )
            block_mask = torch.zeros(num_blocks, device=device)
            block_mask[keep] = 1.0
            block_mask = block_mask.view(num_blocks_h, num_blocks_w)
            full_mask = (
                block_mask
                .repeat_interleave(block_size, dim=0)
                .repeat_interleave(block_size, dim=1)
            )
            masks.append(full_mask)

        # Shape: [B, 1, H, W]  — caller broadcasts over C channels
        return torch.stack(masks, dim=0).unsqueeze(1)

    def forward(self, template, search, is_distill=False):

        # ✅ FIX: always define mask
        mask = None

        # ---------- Mask Generation ----------
        if self.training and not is_distill:

            if self.randomMask:
                mask = self.masking(template, 16, 0.5, template.device)
            else:
                mask = self.masking_CoxProcess_pytorch(
                    template,
                    mask_ratio=0.5,
                    block_size=16,
                )

        # ---------- Backbone ----------
        x, aux_dict = self.backbone(z=template, x=search)

        # ---------- Similarity Loss ----------
        if self.training and mask is not None:

            x1, _ = self.backbone(
                z=template * mask,
                x=search,
            )

            sim_loss = torch.nn.functional.mse_loss(
                x1[:, :self.feat_len_t],
                x[:, :self.feat_len_t].detach(),
            )
        else:
            sim_loss = 0

        # ---------- Head ----------
        feat_last = x[-1] if isinstance(x, list) else x

        out = self.forward_head(feat_last)

        out.update(aux_dict)
        out["backbone_feat"] = x
        out["sim_loss"] = sim_loss

        return out

    # =========================================================
    # -------------------- HEAD FORWARD ------------------------
    # =========================================================

    def forward_head(self, cat_feature, gt_score_map=None):

        enc_opt = cat_feature[:, -self.feat_len_s:]

        opt = enc_opt.unsqueeze(-1).permute(0, 3, 2, 1).contiguous()

        bs, Nq, C, HW = opt.size()

        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":

            pred_box, score_map = self.box_head(opt_feat, True)

            outputs_coord = box_xyxy_to_cxcywh(pred_box)

            return {
                "pred_boxes": outputs_coord.view(bs, Nq, 4),
                "score_map": score_map,
            }

        elif self.head_type == "CENTER":

            score_map_ctr, bbox, size_map, offset_map = self.box_head(
                opt_feat,
                gt_score_map,
            )

            return {
                "pred_boxes": bbox.view(bs, Nq, 4),
                "score_map": score_map_ctr,
                "size_map": size_map,
                "offset_map": offset_map,
            }

        else:
            raise NotImplementedError


# =========================================================
# ---------------- MODEL BUILDER ---------------------------
# =========================================================

def build_ortrack(cfg, training=True):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "../../../pretrained_models")

    pretrained = ""
    if cfg.MODEL.PRETRAIN_FILE and "ORTrack" not in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)

    # -------- Backbone --------
    if cfg.MODEL.BACKBONE.TYPE == "deit_tiny_patch16_224":
        backbone = deit_tiny_patch16_224(num_classes=0, pretrained=True)

    elif cfg.MODEL.BACKBONE.TYPE == "deit_tiny_distilled_patch16_224":
        backbone = deit_tiny_patch16_224_distill(num_classes=0, pretrained=True)

    elif cfg.MODEL.BACKBONE.TYPE == "vit_tiny_patch16_224":
        backbone = vit_tiny_patch16_224(num_classes=0, pretrained=True)

    elif cfg.MODEL.BACKBONE.TYPE == "vit_tiny_distilled_patch16_224":
        backbone = vit_tiny_distilled_patch16_224(num_classes=0, pretrained=True)

    elif cfg.MODEL.BACKBONE.TYPE == "eva02_tiny_patch14_224":
        backbone = eva02_tiny_patch14_224(num_classes=0, pretrained=True)

    elif cfg.MODEL.BACKBONE.TYPE == "eva02_tiny_distilled_patch14_224":
        backbone = eva02_tiny_patch14_224_distill(num_classes=0, pretrained=True)

    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim

    box_head = build_box_head(cfg, hidden_dim)

    model = ORTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if "ORTrack" in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        model.load_state_dict(checkpoint["net"], strict=False)
        print("Load pretrained model from:", cfg.MODEL.PRETRAIN_FILE)

    return model