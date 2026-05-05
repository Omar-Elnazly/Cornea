import json
import random
import cv2
import sys
import os
import re
import numpy as np
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ORTrack'))

from lib.test.tracker.ortrack import ORTrack
from lib.config.ortrack.config import cfg, update_config_from_file


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISM SETUP  ← moved into a function, called before EVERY sequence
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

def set_seeds():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)          # for multi-GPU safety
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # ← the key missing line
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required by PyTorch ≥1.11

set_seeds()  # once at startup too

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
ORTRACK_DIR = os.path.join(BASE_DIR, 'ORTrack')

MANIFEST_PATH = os.path.join(BASE_DIR, 'metadata', 'contestant_manifest_val30.json')
DATA_ROOT     = os.path.join(BASE_DIR, 'data')

CHECKPOINT_PATH = os.path.join(
    ORTRACK_DIR, 'output', 'checkpoints', 'train',
    'ortrack', 'deit_tiny_patch16_224', 'ORTrack_ep0300.pth'
)

W1 = 0.6
W2 = 0.4
# ─────────────────────────────────────────────────────────────────────────────


def load_tracker():
    config_name = 'deit_tiny_patch16_224'
    config_path = os.path.join(
        ORTRACK_DIR, 'experiments', 'ortrack', f'{config_name}.yaml'
    )
    update_config_from_file(config_path)

    from easydict import EasyDict
    params = EasyDict()
    params.cfg             = cfg
    params.search_factor   = cfg.TEST.SEARCH_FACTOR
    params.search_size     = cfg.TEST.SEARCH_SIZE
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size   = cfg.TEST.TEMPLATE_SIZE
    params.checkpoint      = CHECKPOINT_PATH
    params.debug           = 0
    params.use_gpu         = True
    params.vis_attn        = 0
    params.save_all_boxes  = False

    return ORTrack(params, config_name)


def parse_annotation(path):
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                boxes.append([0, 0, 0, 0])
                continue
            parts = re.split(r'[,\s\t]+', line)
            parts = [p for p in parts if p]
            if len(parts) < 4:
                boxes.append([0, 0, 0, 0])
            else:
                try:
                    boxes.append([float(p) for p in parts[:4]])
                except ValueError:
                    boxes.append([0, 0, 0, 0])
    return np.array(boxes, dtype=np.float32)


def track_sequence(tracker, video_path, init_bbox, n_frames):
    cap = cv2.VideoCapture(video_path)

    # ── Force software decoding to avoid hardware decoder non-determinism ──
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)  # ← NEW
    # Disable OpenCV's internal thread pool to prevent frame-read races
    cv2.setNumThreads(0)  # ← NEW

    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            tracker.initialize(frame_rgb, {'init_bbox': init_bbox})
            results.append(init_bbox)
        else:
            output = tracker.track(frame_rgb)
            pred_box = list(output['target_bbox'])
            results.append([float(pred_box[0]), float(pred_box[1]),
                            float(pred_box[2]), float(pred_box[3])])
        frame_idx += 1

    cap.release()
    return np.array(results, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def iou(pred, gt):
    px, py, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx, gy, gw, gh = gt[:, 0],  gt[:, 1],  gt[:, 2],  gt[:, 3]
    ix1 = np.maximum(px, gx);   iy1 = np.maximum(py, gy)
    ix2 = np.minimum(px+pw, gx+gw); iy2 = np.minimum(py+ph, gy+gh)
    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    union = pw*ph + gw*gh - inter + 1e-9
    return inter / union


def norm_dist(pred, gt):
    cx_p = pred[:, 0] + pred[:, 2]/2;  cy_p = pred[:, 1] + pred[:, 3]/2
    cx_g = gt[:, 0]   + gt[:, 2]/2;    cy_g = gt[:, 1]   + gt[:, 3]/2
    dist = np.sqrt((cx_p-cx_g)**2 + (cy_p-cy_g)**2)
    diag = np.sqrt(gt[:, 2]**2 + gt[:, 3]**2) + 1e-9
    return dist / diag


def score_sequence(pred_boxes, gt_boxes):
    visible = np.array(
        [(i > 0) and not (gt_boxes[i, 2] == 0 and gt_boxes[i, 3] == 0)
         for i in range(len(gt_boxes))], dtype=bool)
    if visible.sum() == 0:
        return 0.0, 0.0
    p = pred_boxes[visible];  g = gt_boxes[visible]
    iou_vals  = iou(p, g)
    nd_vals   = norm_dist(p, g)
    auc       = np.mean([np.mean(iou_vals >= t)  for t in np.linspace(0, 1,   21)])
    norm_prec = np.mean([np.mean(nd_vals   <  t) for t in np.linspace(0, 0.5, 51)])
    return float(auc), float(norm_prec)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    sequences = manifest['train']
    total     = len(sequences)
    print(f"Evaluating on {total} validation sequences (held-out, never trained on)\n")

    per_seq = {}

    for idx, (seq_id, seq_info) in enumerate(sequences.items()):
        video_path = os.path.join(DATA_ROOT, seq_info['video_path'].replace('/', os.sep))
        ann_path   = os.path.join(DATA_ROOT, seq_info['annotation_path'].replace('/', os.sep))
        n_frames   = seq_info['n_frames']

        gt_boxes  = parse_annotation(ann_path)
        init_bbox = [int(gt_boxes[0, 0]), int(gt_boxes[0, 1]),
                     int(gt_boxes[0, 2]), int(gt_boxes[0, 3])]

        print(f"[{idx+1}/{total}] {seq_id} ({n_frames} frames) | init: {init_bbox}")

        set_seeds()  

        tracker    = load_tracker()
        pred_boxes = track_sequence(tracker, video_path, init_bbox, n_frames)

        min_len    = min(len(pred_boxes), len(gt_boxes))
        pred_boxes = pred_boxes[:min_len]
        gt_boxes   = gt_boxes[:min_len]

        auc, np_val = score_sequence(pred_boxes, gt_boxes)
        per_seq[seq_id] = {'auc': auc, 'norm_precision': np_val}
        print(f"         AUC={auc:.4f}  NormPrec={np_val:.4f}")

    mean_auc = float(np.mean([v['auc']            for v in per_seq.values()]))
    mean_np  = float(np.mean([v['norm_precision'] for v in per_seq.values()]))
    sacc     = W1 * mean_auc + W2 * mean_np

    print("\n" + "=" * 50)
    print("  Local Accuracy Score — ORTrack")
    print("=" * 50)
    print(f"  Sequences scored : {len(per_seq)}")
    print(f"  AUC (Success)    : {mean_auc:.4f}")
    print(f"  NormPrecision    : {mean_np:.4f}")
    print(f"  Sacc = {W1}*AUC + {W2}*NormPrec = {sacc:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()