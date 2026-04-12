"""
local_eval.py
─────────────────────────────────────────────────────────────────────────────
Evaluates HiT-Base locally using a held-out slice of the TRAINING data.
Computes the exact MTC-AIC4 scoring formula:

    Sacc  = 0.6 * AUC  +  0.4 * NormPrecision
    Seff  = 0.25*FLOPs_n + 0.15*Params_n + 0.35*Latency_n + 0.25*Size_n
    Final = Sacc - 0.2 * Seff

Run from F:\\Cornea:
    conda activate HIT
    python local_eval.py

Adjust LOCAL_TEST_RATIO to control how many sequences are used for local test.
"""

import json
import cv2
import sys
import os
import time
import torch
import numpy as np
import random

# ── CONFIG ────────────────────────────────────────────────────────────────────
LOCAL_TEST_RATIO  = 0.15      # use 15% of train sequences (~38 seqs) as local test
RANDOM_SEED       = 42
CHECKPOINT_PATH   = r'F:\Cornea\output\checkpoints\train\HiT\HiT_Base\HiT_Base.pth'
MANIFEST_PATH     = r'F:\Cornea\metadata\contestant_manifest.json'
DATA_ROOT         = r'F:\Cornea\data'

# Template update settings (set USE_TEMPLATE_UPDATE = False to test baseline)
USE_TEMPLATE_UPDATE   = False   # ← flip to True to test template update
UPDATE_INTERVAL       = 20
MAX_UPDATES           = 10
STABILITY_IOU_THRESH  = 0.85    # box must overlap >=85% with prev box to be "stable"
STABILITY_WINDOW      = 5       # must be stable for this many consecutive frames

# Efficiency budget maximums (from competition rules)
MAX_FLOPS_G   = 30.0     # GFLOPs
MAX_PARAMS_M  = 50.0     # Million params
MAX_LATENCY   = 30.0     # ms
MAX_SIZE_MB   = 500.0    # MB  (0.5 GB)

# ── PROFILED EFFICIENCY VALUES (from profile_model_hit.py --config HiT_Base) ──
# These are the TRUE model-only numbers, not polluted by cv2/video decode overhead.
# Re-run: python tracking/profile_model_hit.py --config HiT_Base  to update.
PROFILED_FLOPS_G   = 4.346 * 2   # MACs → FLOPs:  4.346 GMACs × 2 = 8.692 GFLOPs
PROFILED_PARAMS_M  = 42.14        # from profile script
PROFILED_LATENCY   = 11.53        # ms — pure model inference on your GPU

# ── PATH SETUP ────────────────────────────────────────────────────────────────
sys.path.insert(0, r'F:\Cornea\HiT')
from lib.test.tracker.HiT import HiT as HiTTracker
from lib.config.HiT.config import cfg, update_config_from_file


# ─────────────────────────────────────────────────────────────────────────────
# TRACKER LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_tracker():
    config_path = os.path.join(r'F:\Cornea\HiT', 'experiments', 'HiT', 'HiT_Base.yaml')
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

    tracker = HiTTracker(params, 'HiT_Base')
    return tracker


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION PARSING
# ─────────────────────────────────────────────────────────────────────────────
def parse_annotations(ann_path):
    """Returns list of [x, y, w, h] per frame. [0,0,0,0] = invisible."""
    boxes = []
    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace('\t', ',').replace(' ', ',').split(',')
            parts = [p for p in parts if p]
            if len(parts) >= 4:
                boxes.append([float(p) for p in parts[:4]])
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# TRACKING
# ─────────────────────────────────────────────────────────────────────────────
def box_iou(box_a, box_b):
    """IoU between two [x, y, w, h] boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_a[2]*box_a[3] + box_b[2]*box_b[3] - inter
    return inter / union if union > 0 else 0.0


def is_stable(recent_boxes, iou_thresh, window):
    """True if the last `window` boxes are all consistent with each other."""
    if len(recent_boxes) < window:
        return False
    current = recent_boxes[-1]
    for prev in recent_boxes[-window:-1]:
        if box_iou(current, prev) < iou_thresh:
            return False
    return True


def box_is_valid(box, frame_h, frame_w):
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False
    cx, cy = x + w / 2, y + h / 2
    return 0 <= cx <= frame_w and 0 <= cy <= frame_h


def track_sequence(tracker, video_path, init_bbox, n_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [[0, 0, 0, 0]] * n_frames

    results     = []
    frame_idx   = 0
    last_update = 0
    n_updates   = 0
    recent_boxes = []
    frame_h = frame_w = None
    latencies   = []

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()

        if frame_idx == 0:
            frame_h, frame_w = frame.shape[:2]
            tracker.initialize(frame_rgb, {'init_bbox': init_bbox})
            results.append(list(init_bbox))
            recent_boxes.append(list(init_bbox))
        else:
            output   = tracker.track(frame_rgb)
            pred_box = output['target_bbox']
            pred_box = [int(pred_box[0]), int(pred_box[1]),
                        int(pred_box[2]), int(pred_box[3])]
            results.append(pred_box)
            recent_boxes.append(pred_box)

            if len(recent_boxes) > STABILITY_WINDOW + 1:
                recent_boxes.pop(0)

            if USE_TEMPLATE_UPDATE:
                gap_ok    = (frame_idx - last_update) >= UPDATE_INTERVAL
                budget_ok = n_updates < MAX_UPDATES
                valid     = box_is_valid(pred_box, frame_h, frame_w)
                stable    = is_stable(recent_boxes, STABILITY_IOU_THRESH, STABILITY_WINDOW)
                if gap_ok and budget_ok and valid and stable:
                    tracker.initialize(frame_rgb, {'init_bbox': pred_box})
                    last_update = frame_idx
                    n_updates  += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if frame_idx > 0:           # skip frame 0 (includes model loading overhead)
            latencies.append(elapsed_ms)

        frame_idx += 1

    cap.release()

    while len(results) < n_frames:
        results.append([0, 0, 0, 0])

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    return results, avg_latency


# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION  (exact competition formulas)
# ─────────────────────────────────────────────────────────────────────────────
def iou(box_a, box_b):
    """IoU between two [x,y,w,h] boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_x1 = max(ax1, bx1);  inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2);  inter_y2 = min(ay2, by2)
    inter    = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    union = box_a[2]*box_a[3] + box_b[2]*box_b[3] - inter
    return inter / union if union > 0 else 0.0


def center_distance(box_pred, box_gt):
    """Euclidean distance between box centres."""
    cx_p = box_pred[0] + box_pred[2] / 2
    cy_p = box_pred[1] + box_pred[3] / 2
    cx_g = box_gt[0]   + box_gt[2]   / 2
    cy_g = box_gt[1]   + box_gt[3]   / 2
    return np.sqrt((cx_p - cx_g)**2 + (cy_p - cy_g)**2)


def compute_sequence_metrics(preds, gts):
    """
    Compute per-sequence AUC (Success) and NormPrecision,
    matching the competition evaluation protocol.

    Only frames where ground-truth is visible (w>0 and h>0) are counted.
    """
    iou_scores      = []
    norm_dist_scores = []

    for pred, gt in zip(preds, gts):
        # Skip invisible frames
        if gt[2] <= 0 or gt[3] <= 0:
            continue

        iou_val = iou(pred, gt)
        iou_scores.append(iou_val)

        dist = center_distance(pred, gt)
        # Normalise by diagonal of GT box
        diag = np.sqrt(gt[2]**2 + gt[3]**2)
        norm_dist = dist / diag if diag > 0 else 1.0
        norm_dist_scores.append(norm_dist)

    if not iou_scores:
        return 0.0, 0.0, 0.0

    # ── Success / AUC ─────────────────────────────────────────────────────
    # Area under the success plot over thresholds 0..1 (step 0.05)
    thresholds = np.arange(0, 1.05, 0.05)
    success_rates = [np.mean([s >= t for s in iou_scores]) for t in thresholds]
    auc = float(np.mean(success_rates))

    # ── Precision ─────────────────────────────────────────────────────────
    prec_thresholds = np.arange(0, 0.55, 0.05)   # normalised distance thresholds
    norm_prec_rates = [np.mean([d <= t for d in norm_dist_scores])
                       for t in prec_thresholds]
    norm_precision = float(np.mean(norm_prec_rates))

    # ── Raw precision at 20px (informational only) ─────────────────────────
    precision_20 = float(np.mean([center_distance(p, g) <= 20
                                  for p, g in zip(preds, gts)
                                  if g[2] > 0 and g[3] > 0]))

    return auc, norm_precision, precision_20


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENCY METRICS
# ─────────────────────────────────────────────────────────────────────────────
def measure_efficiency(tracker_obj):
    """
    Returns efficiency metrics using the values from HiT's own profiler
    (tracking/profile_model_hit.py --config HiT_Base).
    This avoids cv2/video-decode overhead polluting the latency number.
    """
    size_bytes = os.path.getsize(CHECKPOINT_PATH)
    size_mb    = size_bytes / (1024 ** 2)
    flops_g    = PROFILED_FLOPS_G
    params_m   = PROFILED_PARAMS_M
    print(f"  [Using profiled values from profile_model_hit.py --config HiT_Base]")
    return flops_g, params_m, size_mb


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SCORE  (exact competition formula)
# ─────────────────────────────────────────────────────────────────────────────
def competition_score(auc, norm_precision, flops_g, params_m,
                       avg_latency_ms, size_mb):
    w1, w2 = 0.6, 0.4
    w3, w4, w5, w6 = 0.25, 0.15, 0.35, 0.25
    lam = 0.2

    sacc = w1 * auc + w2 * norm_precision

    # Normalise efficiency metrics  m' = min(1, m / m_max)
    flops_n   = min(1.0, flops_g        / MAX_FLOPS_G)
    params_n  = min(1.0, params_m       / MAX_PARAMS_M)
    lat_n     = min(1.0, avg_latency_ms / MAX_LATENCY)
    size_n    = min(1.0, size_mb        / MAX_SIZE_MB)

    seff  = w3*flops_n + w4*params_n + w5*lat_n + w6*size_n
    final = sacc - lam * seff
    return sacc, seff, final


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  MTC-AIC4  —  Local Evaluation")
    print(f"  Checkpoint : {CHECKPOINT_PATH}")
    print(f"  Template update: {USE_TEMPLATE_UPDATE}")
    print("=" * 65)

    # ── Load manifest and pick local test sequences ───────────────────────
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    train_seqs = list(manifest['train'].items())
    random.seed(RANDOM_SEED)
    random.shuffle(train_seqs)

    n_test = max(1, int(len(train_seqs) * LOCAL_TEST_RATIO))
    test_seqs = train_seqs[:n_test]

    print(f"\nTotal train sequences : {len(train_seqs)}")
    print(f"Using as local test   : {n_test}  (ratio={LOCAL_TEST_RATIO})")
    print()

    # ── Load tracker ──────────────────────────────────────────────────────
    tracker = load_tracker()
    print("Tracker loaded.\n")

    # ── Measure static efficiency metrics ─────────────────────────────────
    flops_g, params_m, size_mb = measure_efficiency(tracker)
    print(f"Model efficiency:")
    print(f"  FLOPs    : {flops_g:.2f} GFLOPs  (budget: {MAX_FLOPS_G})")
    print(f"  Params   : {params_m:.2f} M       (budget: {MAX_PARAMS_M} M)")
    print(f"  Size     : {size_mb:.1f} MB      (budget: {MAX_SIZE_MB} MB)")
    print()

    # ── Run tracking on each local-test sequence ──────────────────────────
    all_auc, all_norm_prec, all_prec20 = [], [], []
    all_latencies = []
    failed = []

    for i, (seq_id, seq_info) in enumerate(test_seqs):
        video_path = os.path.join(DATA_ROOT, seq_info['video_path'])
        ann_path   = os.path.join(DATA_ROOT, seq_info['annotation_path'])
        n_frames   = seq_info['n_frames']

        if not os.path.isfile(video_path):
            print(f"  [{i+1}/{n_test}] SKIP (video missing): {seq_id}")
            failed.append(seq_id)
            continue

        gts = parse_annotations(ann_path)
        if not gts:
            print(f"  [{i+1}/{n_test}] SKIP (no annotations): {seq_id}")
            failed.append(seq_id)
            continue

        # First visible frame annotation as init bbox
        init_bbox = None
        for box in gts:
            if box[2] > 0 and box[3] > 0:
                init_bbox = [int(b) for b in box]
                break
        if init_bbox is None:
            print(f"  [{i+1}/{n_test}] SKIP (all invisible): {seq_id}")
            failed.append(seq_id)
            continue

        preds, avg_lat = track_sequence(tracker, video_path, init_bbox, n_frames)
        auc, norm_prec, prec20 = compute_sequence_metrics(preds, gts)

        all_auc.append(auc)
        all_norm_prec.append(norm_prec)
        all_prec20.append(prec20)
        all_latencies.append(avg_lat)

        print(f"  [{i+1}/{n_test}] {seq_id:<40s}  "
              f"AUC={auc:.4f}  NormPrec={norm_prec:.4f}  "
              f"Lat={avg_lat:.1f}ms")

    if not all_auc:
        print("\nNo sequences evaluated. Check your data paths.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────
    mean_auc       = float(np.mean(all_auc))
    mean_norm_prec = float(np.mean(all_norm_prec))
    mean_prec20    = float(np.mean(all_prec20))
    mean_latency   = float(np.mean(all_latencies))

    sacc, seff, final = competition_score(
        mean_auc, mean_norm_prec,
        flops_g, params_m, PROFILED_LATENCY, size_mb
    )

    # ── Print results ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Sequences evaluated : {len(all_auc)}  (failed/skipped: {len(failed)})")
    print()
    print(f"  Tracking accuracy:")
    print(f"    AUC (Success)      : {mean_auc:.4f}")
    print(f"    NormPrecision      : {mean_norm_prec:.4f}")
    print(f"    Precision @20px    : {mean_prec20:.4f}  (informational)")
    print()
    print(f"  Efficiency:")
    print(f"    FLOPs              : {flops_g:.2f} GFLOPs")
    print(f"    Params             : {params_m:.2f} M")
    print(f"    Avg latency        : {PROFILED_LATENCY:.2f} ms  (from profile_model_hit.py)")
    print(f"    Model size         : {size_mb:.1f} MB")
    print()
    print(f"  ── Score breakdown ──────────────────────────────────────")
    print(f"    Sacc  = 0.6×{mean_auc:.4f} + 0.4×{mean_norm_prec:.4f} = {sacc:.4f}")
    print(f"    Seff  = {seff:.4f}")
    print(f"    Final = {sacc:.4f} - 0.2×{seff:.4f} = {final:.4f}")
    print()
    print(f"  ★ ESTIMATED COMPETITION SCORE : {final:.4f}")
    print("=" * 65)

    if failed:
        print(f"\nSkipped sequences: {failed}")


if __name__ == '__main__':
    main()