import os
import sys
import cv2

BASE_DIR = os.path.dirname(__file__)
ORTRACK_DIR = os.path.join(BASE_DIR, 'ORTrack')

sys.path.insert(0, ORTRACK_DIR)

import time
import torch
import numpy as np
from thop import profile
from easydict import EasyDict
from lib.config.ortrack.config import cfg, update_config_from_file
from lib.test.tracker.ortrack import ORTrack

#CONFIGURATION
config_path = os.path.join(
    ORTRACK_DIR,
    'experiments',
    'ortrack',
    'deit_tiny_patch16_224.yaml'
)

update_config_from_file(config_path)

checkpoint_path = os.path.join(
    ORTRACK_DIR,
    'output',
    'checkpoints',
    'train',
    'ortrack',
    'Train_V1.3',
    'ORTrack_ep0020.pth'
)

params = EasyDict()
params.cfg             = cfg
params.search_factor   = cfg.TEST.SEARCH_FACTOR
params.search_size     = cfg.TEST.SEARCH_SIZE
params.template_factor = cfg.TEST.TEMPLATE_FACTOR
params.template_size   = cfg.TEST.TEMPLATE_SIZE
params.checkpoint      = checkpoint_path
params.debug           = 0
params.use_gpu         = torch.cuda.is_available()
params.vis_attn        = 0
params.save_all_boxes  = False

tracker = ORTrack(params, 'deit_tiny_patch16_224')
model   = tracker.network
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = model.to(device)

print(f"\n{'='*50}")
print(f"Device: {device}")
print(f"{'='*50}\n")

#PARAMETERS
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   PARAMETERS")
print(f"   Total:     {total_params/1e6:.2f} M")
print(f"   Trainable: {trainable_params/1e6:.2f} M")
print(f"   Budget:    50.00 M")
print(f"   Status:    {'OK' if total_params/1e6 < 50 else 'OVER BUDGET'}\n")

#FLOPS
template_size = cfg.TEST.TEMPLATE_SIZE
search_size   = cfg.TEST.SEARCH_SIZE

template = torch.randn(1, 3, template_size, template_size).to(device)
search   = torch.randn(1, 3, search_size,   search_size).to(device)

try:
    flops, _ = profile(model, inputs=(template, search), verbose=False)
    flops_g  = flops / 1e9
    print(f"   FLOPs")
    print(f"   Measured: {flops_g:.2f} GFLOPs")
    print(f"   Budget:   30.00 GFLOPs")
    print(f"   Status:   {'OK' if flops_g < 30 else 'OVER BUDGET'}\n")
except Exception as e:
    print(f"   FLOPs — measurement failed: {e}\n")
    flops_g = 0

#LATENCY
print(f"   LATENCY")

WARMUP  = 20
REPEATS = 100

with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(template, search)

if device == 'cuda':
    torch.cuda.synchronize()

times = []
with torch.no_grad():
    for _ in range(REPEATS):
        start = time.perf_counter()
        _ = model(template, search)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

avg_latency = np.mean(times)
std_latency = np.std(times)

print(f"   Average:  {avg_latency:.2f} ms  (±{std_latency:.2f} ms)")
print(f"   Budget:   30.00 ms")
print(f"   Status:   {'OK' if avg_latency < 30 else 'OVER BUDGET'}\n")

#MODEL SIZE
model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
model_size_gb = model_size_mb / 1024

print(f"   MODEL SIZE")
print(f"   Size:    {model_size_mb:.2f} MB  ({model_size_gb:.3f} GB)")
print(f"   Budget:  500.00 MB  (0.5 GB)")
print(f"   Status:  {'OK' if model_size_mb < 500 else 'OVER BUDGET'}\n")

#EFFICIENCY SCORE
print(f"{'='*50}")
print(f"  EFFICIENCY SCORE PREVIEW")
print(f"{'='*50}")

flops_norm   = min(1.0, flops_g / 30.0)
params_norm  = min(1.0, (total_params/1e6) / 50.0)
latency_norm = min(1.0, avg_latency / 30.0)
size_norm    = min(1.0, model_size_mb / 500.0)

seff = (0.25 * flops_norm +
        0.15 * params_norm +
        0.35 * latency_norm +
        0.25 * size_norm)

print(f"   FLOPs    normalized: {flops_norm:.4f}  (weight 0.25)")
print(f"   Params   normalized: {params_norm:.4f}  (weight 0.15)")
print(f"   Latency  normalized: {latency_norm:.4f}  (weight 0.35)")
print(f"   Size     normalized: {size_norm:.4f}  (weight 0.25)")
print(f"\n   Seff  = {seff:.4f}")
print(f"   Penalty = 0.2 × {seff:.4f} = {0.2*seff:.4f}")
print(f"\n   (Lower Seff = better. Max penalty possible = 0.20)")