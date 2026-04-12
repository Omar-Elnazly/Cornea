import json
import cv2
import csv
import sys
import os
import torch
import numpy as np

# Add HiT to path
sys.path.insert(0, 'F:\\Cornea\\HiT')

from HiT.lib.test.utils import params
from lib.test.tracker.HiT import HiT as HiTTracker
from lib.config.HiT.config import cfg, update_config_from_file

def load_tracker():
    config_path = os.path.join('HiT', 'experiments', 'HiT', 'HiT_Base.yaml')
    update_config_from_file(config_path)

    from easydict import EasyDict
    checkpoint_path = r'F:\Cornea\output\checkpoints\train\HiT\HiT_Base_finetune\checkpoints\HiT\HiT_Base_finetune\HiT_ep0050.pth.tar'

    params = EasyDict()
    params.cfg = cfg
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.checkpoint = checkpoint_path
    params.debug = 0
    params.use_gpu = True
    params.vis_attn = 0
    params.save_all_boxes = False

    tracker = HiTTracker(params, 'HiT_Base')
    print("HiT-Base loaded successfully.")
    return tracker

def track_sequence(tracker, video_path, init_bbox, n_frames):
    cap = cv2.VideoCapture(video_path)
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
            pred_box = output['target_bbox']
            results.append([
                int(pred_box[0]),
                int(pred_box[1]),
                int(pred_box[2]),
                int(pred_box[3])
            ])

        frame_idx += 1

    cap.release()
    return results

def main():
    # Paths
    manifest_path   = os.path.join('metadata', 'contestant_manifest.json')
    submission_path = os.path.join('metadata', 'sample_submission.csv')
    output_csv      = 'submission.csv'
    data_root       = 'data'

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load sample submission row order
    submission_rows = []
    with open(submission_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            submission_rows.append(row['id'])

    print(f"Total frames to predict: {len(submission_rows)}")

    # Load tracker
    tracker = load_tracker()

    # Run inference on every public_lb sequence
    all_predictions = {}

    public_lb = manifest.get('public_lb', {})
    total_seqs = len(public_lb)

    for idx, (seq_id, seq_info) in enumerate(public_lb.items()):
        video_path = os.path.join(data_root, seq_info['video_path'])
        n_frames   = seq_info['n_frames']

        print(f"[{idx+1}/{total_seqs}] Tracking: {seq_id} ({n_frames} frames)")

        # Get first frame bounding box from annotation
        annotation_path = seq_info.get('annotation_path')
        if annotation_path:
            ann_full = os.path.join(data_root, annotation_path)
            with open(ann_full, 'r') as f:
                first_line = f.readline().strip()
            parts = first_line.replace('\t', ',').split(',')
            init_bbox = [int(float(p)) for p in parts[:4]]
        else:
            # Test sequences: read first frame dimensions as fallback
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            h, w = frame.shape[:2]
            init_bbox = [w//4, h//4, w//2, h//2]

        # Track
        preds = track_sequence(tracker, video_path, init_bbox, n_frames)

        # Store predictions
        for i, box in enumerate(preds):
            row_id = f"{seq_id}_{i}"
            all_predictions[row_id] = box

    # Write submission CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'w', 'h'])
        for row_id in submission_rows:
            if row_id in all_predictions:
                box = all_predictions[row_id]
                writer.writerow([row_id, box[0], box[1], box[2], box[3]])
            else:
                writer.writerow([row_id, 0, 0, 0, 0])

    print(f"\nDone! Submission saved to: {output_csv}")
    print(f"Total rows written: {len(submission_rows)}")

if __name__ == '__main__':
    main()