"""
diagnose_hit.py
───────────────────────────────────────────────────────────────
Runs HiT on the FIRST training sequence for 50 frames and prints
exactly what track() returns each frame — keys, score values, etc.

Run from F:\\Cornea:
    conda activate HIT
    python diagnose_hit.py
"""

import json, cv2, sys, os, torch
sys.path.insert(0, r'F:\Cornea\HiT')

from lib.test.tracker.HiT import HiT as HiTTracker
from lib.config.HiT.config import cfg, update_config_from_file
from easydict import EasyDict

MANIFEST  = r'F:\Cornea\metadata\contestant_manifest.json'
DATA_ROOT = r'F:\Cornea\data'
CKPT      = r'F:\Cornea\output\checkpoints\train\HiT\HiT_Base\HiT_Base.pth'
N_FRAMES  = 50   # only look at first 50 frames

# ── Load tracker ─────────────────────────────────────────────
update_config_from_file(r'F:\Cornea\HiT\experiments\HiT\HiT_Base.yaml')
params = EasyDict()
params.cfg = cfg
params.search_factor  = cfg.TEST.SEARCH_FACTOR
params.search_size    = cfg.TEST.SEARCH_SIZE
params.template_factor = cfg.TEST.TEMPLATE_FACTOR
params.template_size  = cfg.TEST.TEMPLATE_SIZE
params.checkpoint     = CKPT
params.debug          = 0
params.use_gpu        = True
params.vis_attn       = 0
params.save_all_boxes = False

tracker = HiTTracker(params, 'HiT_Base')
print("Tracker loaded.\n")

# ── Pick first training sequence ──────────────────────────────
with open(MANIFEST) as f:
    manifest = json.load(f)

seq_id, seq_info = next(iter(manifest['train'].items()))
video_path = os.path.join(DATA_ROOT, seq_info['video_path'])
ann_path   = os.path.join(DATA_ROOT, seq_info['annotation_path'])

print(f"Sequence : {seq_id}")
print(f"Video    : {video_path}")
print()

# Read first annotation line as init bbox
with open(ann_path) as f:
    first_line = f.readline().strip()
parts     = first_line.replace('\t',',').replace(' ',',').split(',')
parts     = [p for p in parts if p]
init_bbox = [int(float(p)) for p in parts[:4]]
print(f"Init bbox: {init_bbox}\n")

# ── Run tracking and print output each frame ──────────────────
cap = cv2.VideoCapture(video_path)
print(f"{'Frame':<6} {'Keys in output':<35} {'target_bbox':<30} {'score'}")
print("-" * 100)

for i in range(N_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if i == 0:
        tracker.initialize(frame_rgb, {'init_bbox': init_bbox})
        print(f"{i:<6} {'(initialize — no output)'}")
        continue

    output = tracker.track(frame_rgb)

    keys      = list(output.keys())
    bbox      = output.get('target_bbox', 'N/A')
    score_raw = output.get('score', 'NOT PRESENT')

    # Try to get a scalar from score
    if isinstance(score_raw, torch.Tensor):
        score_val = f"Tensor shape={list(score_raw.shape)} val={score_raw.flatten()[0].item():.4f}"
    elif isinstance(score_raw, float):
        score_val = f"{score_raw:.4f}"
    elif score_raw == 'NOT PRESENT':
        score_val = "NOT PRESENT"
    else:
        score_val = str(score_raw)

    print(f"{i:<6} {str(keys):<35} {str([round(b) for b in bbox]):<30} {score_val}")

cap.release()
print()
print("─" * 100)
print("DIAGNOSIS COMPLETE")
print()
print("What to look for:")
print("  1. Is 'score' present in keys?")
print("  2. Does score vary frame to frame, or is it always the same value?")
print("  3. What shape is the score tensor?")
print("  4. Are there other useful keys (e.g. 'conf', 'peak_score') ?")