"""
inference_template_update.py
─────────────────────────────────────────────────────────────────────────────
HiT-Base inference with smart template update.

KEY FINDING: HiT track() only returns {'target_bbox'} — no score/confidence.
So we compute our own confidence from BOX STABILITY:
  - Compare current predicted box vs previous predicted box
  - If IoU between them is high → tracker is stable → safe to update template
  - If IoU is low → tracker jumped → do NOT update (it's probably lost)

Run from F:\\Cornea:
    conda activate HIT
    python inference_template_update.py
"""

import json
import cv2
import csv
import sys
import os
import numpy as np

sys.path.insert(0, 'F:\\Cornea\\HiT')

from lib.test.tracker.HiT import HiT as HiTTracker
from lib.config.HiT.config import cfg, update_config_from_file

# ─────────────────────────────────────────────────────────────
# TEMPLATE UPDATE SETTINGS
# ─────────────────────────────────────────────────────────────
UPDATE_INTERVAL       = 20     # update at most once every N frames
MAX_UPDATES           = 10     # safety cap: never update more than this per sequence
STABILITY_IOU_THRESH  = 0.85   # box must overlap >=85% with previous box to be "stable"
STABILITY_WINDOW      = 5      # box must be stable for this many consecutive frames
# ─────────────────────────────────────────────────────────────


def load_tracker():
    config_path = os.path.join('HiT', 'experiments', 'HiT', 'HiT_Base.yaml')
    update_config_from_file(config_path)

    from easydict import EasyDict
    checkpoint_path = r'F:\Cornea\output\checkpoints\train\HiT\HiT_Base\HiT_Base.pth'

    params = EasyDict()
    params.cfg             = cfg
    params.search_factor   = cfg.TEST.SEARCH_FACTOR
    params.search_size     = cfg.TEST.SEARCH_SIZE
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size   = cfg.TEST.TEMPLATE_SIZE
    params.checkpoint      = checkpoint_path
    params.debug           = 0
    params.use_gpu         = True
    params.vis_attn        = 0
    params.save_all_boxes  = False

    tracker = HiTTracker(params, 'HiT_Base')
    print("HiT-Base loaded successfully.")
    return tracker


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


def box_is_valid(box, frame_h, frame_w):
    """Box must have positive area and centre inside frame."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False
    cx, cy = x + w / 2, y + h / 2
    return 0 <= cx <= frame_w and 0 <= cy <= frame_h


def is_stable(recent_boxes, iou_thresh, window):
    """
    Returns True if the last `window` boxes all have IoU >= iou_thresh
    with the current (last) box. This means the tracker has been
    consistently predicting the same region — it's locked on.
    """
    if len(recent_boxes) < window:
        return False
    current = recent_boxes[-1]
    for prev in recent_boxes[-window:-1]:
        if box_iou(current, prev) < iou_thresh:
            return False
    return True


def track_sequence(tracker, video_path, init_bbox, n_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return [[0, 0, 0, 0]] * n_frames

    results      = []
    frame_idx    = 0
    last_update  = 0
    n_updates    = 0
    recent_boxes = []
    frame_h = frame_w = None

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            gap_ok    = (frame_idx - last_update) >= UPDATE_INTERVAL
            budget_ok = n_updates < MAX_UPDATES
            valid     = box_is_valid(pred_box, frame_h, frame_w)
            stable    = is_stable(recent_boxes, STABILITY_IOU_THRESH, STABILITY_WINDOW)

            if gap_ok and budget_ok and valid and stable:
                tracker.initialize(frame_rgb, {'init_bbox': pred_box})
                last_update = frame_idx
                n_updates  += 1
                print(f"    Template updated at frame {frame_idx} (update #{n_updates})")

        frame_idx += 1

    cap.release()

    while len(results) < n_frames:
        results.append([0, 0, 0, 0])

    return results


def main():
    manifest_path   = os.path.join('metadata', 'contestant_manifest.json')
    submission_path = os.path.join('metadata', 'sample_submission.csv')
    output_csv      = 'submission_template_update.csv'
    data_root       = 'data'

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    submission_rows = []
    with open(submission_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            submission_rows.append(row['id'])

    print(f"Total frames to predict: {len(submission_rows)}")
    print(f"Settings: interval={UPDATE_INTERVAL}, max_updates={MAX_UPDATES}, "
          f"stability_iou={STABILITY_IOU_THRESH}, window={STABILITY_WINDOW}")

    tracker    = load_tracker()
    public_lb  = manifest.get('public_lb', {})
    total_seqs = len(public_lb)
    all_predictions = {}

    for idx, (seq_id, seq_info) in enumerate(public_lb.items()):
        video_path = os.path.join(data_root, seq_info['video_path'])
        n_frames   = seq_info['n_frames']

        print(f"\n[{idx+1}/{total_seqs}] {seq_id}  ({n_frames} frames)")

        annotation_path = seq_info.get('annotation_path')
        if annotation_path:
            ann_full = os.path.join(data_root, annotation_path)
            with open(ann_full, 'r') as f:
                first_line = f.readline().strip()
            parts     = first_line.replace('\t', ',').replace(' ', ',').split(',')
            parts     = [p for p in parts if p]
            init_bbox = [int(float(p)) for p in parts[:4]]
        else:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                h, w      = frame.shape[:2]
                init_bbox = [w // 4, h // 4, w // 2, h // 2]
            else:
                init_bbox = [0, 0, 100, 100]

        preds = track_sequence(tracker, video_path, init_bbox, n_frames)

        for i, box in enumerate(preds):
            all_predictions[f"{seq_id}_{i}"] = box

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'w', 'h'])
        for row_id in submission_rows:
            box = all_predictions.get(row_id, [0, 0, 0, 0])
            writer.writerow([row_id, box[0], box[1], box[2], box[3]])

    print(f"\nDone! Saved to: {output_csv}")
    print(f"Total rows written: {len(submission_rows)}")


if __name__ == '__main__':
    main()