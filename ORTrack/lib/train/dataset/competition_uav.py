import os
import cv2
import numpy as np
import torch
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import jpeg4py_loader_w_failsafe
import random


# ── UAV-Specific Augmentations ────────────────────────────────

def simulate_altitude_change(frame, bbox, scale_range=(0.5, 1.0)):
    """Simulate UAV altitude change by rescaling frame."""
    scale = random.uniform(*scale_range)
    h, w  = frame.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h == 0 or new_w == 0:
        return frame, bbox
    resized = cv2.resize(frame, (new_w, new_h))
    pad_h = (h - new_h) // 2
    pad_w = (w - new_w) // 2
    padded = np.zeros_like(frame)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    new_bbox = [x*scale + pad_w, y*scale + pad_h, bw*scale, bh*scale]
    return padded, new_bbox

def simulate_camera_shake(frame, bbox, max_jitter=12):
    """Simulate UAV camera vibration/wind shake."""
    h, w = frame.shape[:2]
    jx   = random.randint(-max_jitter, max_jitter)
    jy   = random.randint(-max_jitter, max_jitter)
    M    = np.float32([[1, 0, jx], [0, 1, jy]])
    shifted = cv2.warpAffine(frame, M, (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return shifted, [x+jx, y+jy, bw, bh]

def apply_motion_blur(frame, max_kernel=7):
    """Simulate motion blur from fast target or fast drone."""
    kernel_size = random.choice([3, 5, 7])
    direction   = random.choice(['h', 'v', 'd'])
    kernel      = np.zeros((kernel_size, kernel_size))
    if direction == 'h':
        kernel[kernel_size//2, :] = 1.0 / kernel_size
    elif direction == 'v':
        kernel[:, kernel_size//2] = 1.0 / kernel_size
    else:
        np.fill_diagonal(kernel, 1.0 / kernel_size)
    return cv2.filter2D(frame, -1, kernel)

def simulate_small_target(frame, bbox, zoom_range=(1.0, 2.5)):
    """Simulate small/distant target by zooming into search region."""
    zoom  = random.uniform(*zoom_range)
    h, w  = frame.shape[:2]
    new_h = max(1, int(h / zoom))
    new_w = max(1, int(w / zoom))
    y_start = (h - new_h) // 2
    x_start = (w - new_w) // 2
    cropped = frame[y_start:y_start+new_h, x_start:x_start+new_w]
    zoomed  = cv2.resize(cropped, (w, h))
    x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    new_bbox = [(x-x_start)*zoom, (y-y_start)*zoom, bw*zoom, bh*zoom]
    return zoomed, new_bbox


class CompetitionUAV(BaseVideoDataset):
    def __init__(self, root, manifest_path, image_loader=jpeg4py_loader_w_failsafe):
        super().__init__('CompetitionUAV', root, image_loader)
        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        self.sequence_list = []
        for seq_id, seq_info in manifest.get('train', {}).items():
            if seq_info.get('annotation_path') is None:
                continue
            self.sequence_list.append(seq_info)
        print(f"CompetitionUAV: loaded {len(self.sequence_list)} sequences")

    def get_name(self):
        return 'competition_uav'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        seq_info = self.sequence_list[seq_id]
        ann_path = os.path.join(self.root, seq_info['annotation_path'])
        bboxes = []
        visible = []
        with open(ann_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace('\t', ',').replace(' ', ',').split(',')
                parts = [p for p in parts if p]
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                bboxes.append([x, y, w, h])
                visible.append(w > 0 and h > 0)
        bbox      = torch.tensor(bboxes,   dtype=torch.float32)
        valid     = torch.tensor(visible,  dtype=torch.bool)
        visible_t = torch.tensor(visible,  dtype=torch.bool)
        return {'bbox': bbox, 'valid': valid, 'visible': visible_t}

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_info   = self.sequence_list[seq_id]
        video_path = os.path.join(self.root, seq_info['video_path'])
        cap        = cv2.VideoCapture(video_path)
        frames     = []
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((256, 256, 3), dtype=np.uint8)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {k: v[frame_ids] for k, v in anno.items()}

        # ── UAV Augmentations ─────────────────────────────────────
        # frames[0] = template → NEVER augment (it's the reference)
        # frames[1] = search   → augment this one only
        # Only applied during training (when we have at least 2 frames)
        if len(frames) >= 2:
            search = np.array(frames[1], dtype=np.uint8)
            bbox   = anno_frames['bbox'][1]
            # Convert tensor to list if needed
            if hasattr(bbox, 'tolist'):
                bbox = bbox.tolist()
            else:
                bbox = list(bbox)

            try:
                if random.random() < 0.5:
                    search, bbox = simulate_altitude_change(
                        search, bbox, scale_range=(0.5, 1.0))

                if random.random() < 0.4:
                    search, bbox = simulate_camera_shake(
                        search, bbox, max_jitter=12)

                if random.random() < 0.3:
                    search = apply_motion_blur(search, max_kernel=7)

                if random.random() < 0.3:
                    search, bbox = simulate_small_target(
                        search, bbox, zoom_range=(1.0, 2.5))

                frames[1] = search
                # Convert bbox back to tensor to match original dtype
                anno_frames['bbox'][1] = torch.tensor(bbox, dtype=torch.float32)

            except Exception:
                # If augmentation fails for any reason, use original frame
                pass
        # ─────────────────────────────────────────────────────────

        object_meta = {
            'object_class_name': None,
            'motion_class':      None,
            'major_class':       None,
            'root_class':        None,
            'motion_adverb':     None
        }
        return frames, anno_frames, object_meta