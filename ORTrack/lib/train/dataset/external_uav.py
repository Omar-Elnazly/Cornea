import os
import cv2
import numpy as np
import torch
import random
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import jpeg4py_loader_w_failsafe
from lib.train.dataset.competition_uav import (
    simulate_altitude_change,
    simulate_camera_shake,
    apply_motion_blur,
    simulate_small_target,
)


class ExternalUAV(BaseVideoDataset):
    """
    Unified loader for UAV123 and VisDrone-SOT (both parts).
    Returns torch tensors to match ORTrack pipeline expectations.
    """
    def __init__(self, root, image_loader=jpeg4py_loader_w_failsafe,
                 split='train', data_fraction=None):
        super().__init__('ExternalUAV', root, image_loader)
        self.sequence_list = self._build_sequence_list()
        if data_fraction is not None:
            n = int(len(self.sequence_list) * data_fraction)
            self.sequence_list = self.sequence_list[:n]

    def _build_sequence_list(self):
        sequences = []

        # UAV123
        uav123_root = os.path.join(self.root, 'UAV123')
        anno_root   = os.path.join(uav123_root, 'anno', 'UAV123')
        seq_root    = os.path.join(uav123_root, 'data_seq', 'UAV123')
        if os.path.exists(anno_root):
            for anno_file in sorted(os.listdir(anno_root)):
                if not anno_file.endswith('.txt'):
                    continue
                seq_name  = anno_file[:-4]
                seq_dir   = os.path.join(seq_root, seq_name)
                anno_path = os.path.join(anno_root, anno_file)
                if not os.path.exists(seq_dir):
                    continue
                frames = sorted([
                    os.path.join(seq_dir, f)
                    for f in os.listdir(seq_dir) if f.endswith('.jpg')
                ])
                boxes = self._load_anno(anno_path)
                if len(frames) > 0 and len(boxes) > 0:
                    sequences.append({
                        'frames': frames,
                        'boxes':  boxes,
                        'name':   f'UAV123_{seq_name}'
                    })
            print(f'UAV123: {sum(1 for s in sequences if "UAV123_" in s["name"])} sequences loaded')

        # VisDrone-SOT part1 and part2
        for part in ['VisDrone-SOT-part1', 'VisDrone-SOT-part2']:
            vd_root      = os.path.join(self.root, part)
            seq_dir_root = os.path.join(vd_root, 'sequences')
            ann_dir_root = os.path.join(vd_root, 'annotations')
            if not os.path.exists(seq_dir_root):
                print(f'{part}: not found, skipping')
                continue
            part_count = 0
            for seq_name in sorted(os.listdir(seq_dir_root)):
                seq_dir   = os.path.join(seq_dir_root, seq_name)
                anno_path = os.path.join(ann_dir_root, seq_name + '.txt')
                if not os.path.isdir(seq_dir) or not os.path.exists(anno_path):
                    continue
                frames = sorted([
                    os.path.join(seq_dir, f)
                    for f in os.listdir(seq_dir) if f.endswith('.jpg')
                ])
                boxes = self._load_anno(anno_path)
                if len(frames) > 0 and len(boxes) > 0:
                    sequences.append({
                        'frames': frames,
                        'boxes':  boxes,
                        'name':   f'VisDrone_{seq_name}'
                    })
                    part_count += 1
            print(f'{part}: {part_count} sequences loaded')

        print(f'Total ExternalUAV sequences: {len(sequences)}')
        return sequences

    def _load_anno(self, path):
        boxes = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace('\t', ',').replace(' ', ',').split(',')
                parts = [p for p in parts if p]
                if len(parts) >= 4:
                    try:
                        x, y, w, h = (float(parts[0]), float(parts[1]),
                                      float(parts[2]), float(parts[3]))
                        boxes.append([x, y, w, h])
                    except ValueError:
                        continue
        return np.array(boxes, dtype=np.float32)

    def get_name(self):           return 'ExternalUAV'
    def has_class_info(self):     return False
    def has_occlusion_info(self): return False
    def get_num_sequences(self):  return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        """Returns torch tensors — required by ORTrack sampler."""
        boxes = self.sequence_list[seq_id]['boxes']
        valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        return {
            'bbox':    torch.tensor(boxes, dtype=torch.float32),
            'valid':   torch.tensor(valid, dtype=torch.bool),
            'visible': torch.tensor(valid, dtype=torch.bool),
        }

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq    = self.sequence_list[seq_id]
        frames = [self.image_loader(seq['frames'][int(i)]) for i in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # anno values are tensors — index with frame_ids directly
        anno_frames = {k: v[frame_ids] for k, v in anno.items()}

        # UAV Augmentations — search frame (index 1) only
        # Template (index 0) must stay clean — it is the reference image
        if len(frames) >= 2:
            search = np.array(frames[1], dtype=np.uint8)
            bbox   = anno_frames['bbox'][1].tolist()

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
                anno_frames['bbox'][1] = torch.tensor(bbox, dtype=torch.float32)

            except Exception:
                pass  # use original frame if augmentation fails for any reason

        return frames, anno_frames, None
