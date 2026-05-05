import json
import cv2
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ORTrack'))

from lib.test.tracker.ortrack import ORTrack
from lib.config.ortrack.config import cfg, update_config_from_file  


# CONFIGURATION
BASE_DIR = os.path.dirname(__file__)

MANIFEST_PATH = os.path.join(BASE_DIR, 'metadata', 'contestant_manifest.json')
DATA_ROOT     = os.path.join(BASE_DIR, 'data')

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    'ORTrack',
    'output',
    'checkpoints',
    'train',
    'ortrack',
    'Train_v1.3',
    'ORTrack_ep0020.pth'
)

SUBMISSION_TPL = os.path.join(BASE_DIR, 'metadata', 'sample_submission.csv')
OUTPUT_CSV     = os.path.join(BASE_DIR, 'submission_output.csv')


def load_tracker():
    config_name = 'deit_tiny_patch16_224'
    config_path = os.path.join(os.path.dirname(__file__), 'ORTrack', 'experiments', 'ortrack', f'{config_name}.yaml')
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

    tracker = ORTrack(params, config_name)
    print("ORTrack-DeiT loaded successfully.")
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
            results.append([float(pred_box[0]), float(pred_box[1]),
                            float(pred_box[2]), float(pred_box[3])])

        frame_idx += 1

    cap.release()
    return results


def main():
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    submission_rows = []
    with open(SUBMISSION_TPL, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            submission_rows.append(row['id'])
    print(f"Total frames to predict: {len(submission_rows)}")

    all_predictions = {}
    public_lb   = manifest.get('public_lb', {})
    total_seqs  = len(public_lb)
    
    tracker = load_tracker()

    for idx, (seq_id, seq_info) in enumerate(public_lb.items()):
        video_path = os.path.join(DATA_ROOT, seq_info['video_path'].replace('/', os.sep))
        n_frames   = seq_info['n_frames']

        print(f"[{idx+1}/{total_seqs}] Tracking: {seq_id} ({n_frames} frames)")

        annotation_path = seq_info.get('annotation_path')
        if annotation_path:
            ann_full = os.path.join(DATA_ROOT, annotation_path.replace('/', os.sep))
            with open(ann_full, 'r') as f:
                first_line = f.readline().strip()
            parts = first_line.replace('\t', ',').split(',')
            init_bbox = [int(float(p)) for p in parts[:4]]
        else:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            h, w = frame.shape[:2]
            init_bbox = [w//4, h//4, w//2, h//2]

        preds   = track_sequence(tracker, video_path, init_bbox, n_frames)

        for i, box in enumerate(preds):
            row_id = f"{seq_id}_{i}"
            all_predictions[row_id] = box

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'w', 'h'])
        for row_id in submission_rows:
            if row_id in all_predictions:
                box = all_predictions[row_id]
                writer.writerow([row_id, box[0], box[1], box[2], box[3]])
            else:
                writer.writerow([row_id, 0, 0, 0, 0])

    print(f"\nDone! Submission saved to: {OUTPUT_CSV}")
    print(f"Total rows written: {len(submission_rows)}")


if __name__ == '__main__':
    main()