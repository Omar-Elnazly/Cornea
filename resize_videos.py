import os
import cv2
import json
from tqdm import tqdm

data_root = r'F:\Cornea\data'
manifest_path = r'F:\Cornea\metadata\contestant_manifest.json'
output_root = r'F:\Cornea\data_small'
TARGET_SIZE = 640  # resize shortest side to 640px

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

all_sequences = {}
all_sequences.update(manifest['train'])

total = len(all_sequences)
print(f'Resizing {total} sequences to {TARGET_SIZE}p...')

for idx, (seq_id, seq_info) in enumerate(all_sequences.items()):
    video_path = os.path.join(data_root, seq_info['video_path'].replace('/', os.sep))
    out_video_path = os.path.join(output_root, seq_info['video_path'].replace('/', os.sep))
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

    # Skip if already done
    if os.path.exists(out_video_path):
        print(f'[{idx+1}/{total}] Already done: {seq_id}')
        continue

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[{idx+1}/{total}] Cannot open: {seq_id}')
        continue

    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate new size
    if w > h:
        new_w = TARGET_SIZE
        new_h = int(h * TARGET_SIZE / w)
    else:
        new_h = TARGET_SIZE
        new_w = int(w * TARGET_SIZE / h)

    scale_x = new_w / w
    scale_y = new_h / h

    # Write resized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (new_w, new_h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (new_w, new_h))
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Copy and scale annotation file
    if seq_info.get('annotation_path'):
        ann_path = os.path.join(data_root, seq_info['annotation_path'].replace('/', os.sep))
        out_ann_path = os.path.join(output_root, seq_info['annotation_path'].replace('/', os.sep))
        os.makedirs(os.path.dirname(out_ann_path), exist_ok=True)

        with open(ann_path, 'r') as f:
            lines = f.readlines()

        with open(out_ann_path, 'w') as f:
            for line in lines:
                line = line.strip()
                if not line:
                    f.write('\n')
                    continue
                import re
                parts = re.split(r'[,\s\t]+', line)
                parts = [p for p in parts if p]
                if len(parts) < 4:
                    f.write('0,0,0,0\n')
                    continue
                try:
                    x = float(parts[0]) * scale_x
                    y = float(parts[1]) * scale_y
                    w2 = float(parts[2]) * scale_x
                    h2 = float(parts[3]) * scale_y
                    f.write(f'{x:.1f},{y:.1f},{w2:.1f},{h2:.1f}\n')
                except:
                    f.write('0,0,0,0\n')

    print(f'[{idx+1}/{total}] Done: {seq_id} ({w}x{h} -> {new_w}x{new_h})')

print('All videos resized!')
print(f'Resized data saved to: {output_root}')