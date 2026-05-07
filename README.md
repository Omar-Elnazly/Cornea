# Cornea — MTC-AIC4: Efficient Aerial Single-Object Tracking

**Team submission for the MTC-AIC4 Kaggle Competition (Phase I)**
Tracker: [ORTrack](https://github.com/wuyou3474/ORTrack) fine-tuned on aerial UAV datasets.

---

## Model Checkpoint

> **Direct download (Google Drive):**
> [ORTrack_ep0020.pth](https://drive.google.com/file/d/1_4HMxYeH7pjDSaiQ276w39z5Wq4Zd29-/view?usp=sharing)

Place the downloaded checkpoint at:

```
ORTrack/output/checkpoints/train/ortrack/Train_V1.3/ORTrack_ep0020.pth
```

---

## Training

The model was trained in three stages using a Kaggle notebook.

**Training Notebook:** 🔗 [Cornea-3 on Kaggle](https://www.kaggle.com/code/omarelnazly/conrea-3)

### Training Datasets

| #   | Dataset                   | Source                                                                                    |
| --- | ------------------------- | ----------------------------------------------------------------------------------------- |
| 1   | MTC-AIC4 Competition Data | Kaggle Competition                                                                        |
| 2   | VisDrone-SOT (Part 1 & 2) | [VisDrone GitHub](https://github.com/VisDrone/VisDrone-Dataset)                           |
| 3   | UAV123                    | [KAUST Benchmark](https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset) |

### Training Summary

- **Base checkpoint:** `deit_tiny_patch16_224/ortrack_ep0300.pth` (from the official ORTrack repo)
- **Fine-tune config:** `ORTrack/experiments/ortrack/ortrack_finetune.yaml`
- **Final checkpoint:** `Train_V1.3/ORTrack_ep0020.pth`
- Full training details, hyperparameters, and logs are available in the Kaggle notebook linked above.

### Model Performance

**Kaggle Public Leaderboard Score: `0.7538`**

Efficiency metrics measured on **NVIDIA GeForce RTX 3050 Ti Laptop GPU**:

| Metric     | Value                |
| ---------- | -------------------- |
| Parameters | 8.08 M               |
| FLOPs      | 2.39 GFLOPs          |
| Latency    | 12.57 ms (± 0.45 ms) |
| Model Size | 50.13 MB (0.049 GB)  |

---

## Repository Structure

After downloading the dataset and checkpoint, your directory should look like this:

```
Cornea/
├── competition_data/
│   ├── dataset1/
│   ├── dataset2/
│   ├── dataset3/
│   ├── dataset4/
│   └── dataset5/
│       └── <sequence_name>/
│           ├── <sequence_name>.mp4
│           └── annotation.txt
├── metadata/
│   ├── contestant_manifest.json
│   ├── contestant_manifest_train225.json  ← for training
│   ├── contestant_manifest_val30.json     ← for inference_score.py
│   └── sample_submission.csv
├── ORTrack/
│   ├── experiments/
│   │   └── ortrack/
│   │       ├── deit_tiny_patch16_224.yaml
│   │       └── ortrack_finetune.yaml
│   ├── lib/
│   ├── output/
│   │   └── checkpoints/
│   │       └── train/
│   │           └── ortrack/
│   │               └── Train_V1.3/
│   │                   └── ORTrack_ep0020.pth   ← place checkpoint here
│   └── tracking/
├── inference.py
├── inference_score.py
├── measure_efficiency.py
├── resize_videos.py
└── requirements.txt
```

---

## Environment Setup

### Requirements

- Python 3.8
- CUDA-compatible GPU (CUDA 11.8 recommended)
- Windows: [Microsoft Visual Studio 2022](https://visualstudio.microsoft.com/) with **Desktop development with C++** workload

### Step 1 — Clone this repository

```bash
git clone https://github.com/Omar-Elnazly/Cornea
cd Cornea
```

### Step 2 — ORTrack is already included

> **Do not clone the original ORTrack repo separately.**
> This repository already includes the ORTrack folder with all competition-specific modifications applied. Cloning the original would overwrite those changes and break inference.

### Step 3 — Set up the environment

```bash
conda create -n cornea python=3.8 -y
conda activate cornea

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

pip install pycocotools-windows

pip install -r requirements.txt
```

> **Windows note:** `lmdb==1.4.1` and `pycocotools-windows` must be installed separately before `requirements.txt` — they replace the generic `lmdb` and `pycocotools` entries which fail to build from source on Windows.

### Step 4 — Download the checkpoint

Download `ORTrack_ep0020.pth` from the [Google Drive link](#model-checkpoint) above and place it at:

```
ORTrack/output/checkpoints/train/ortrack/Train_V1.3/ORTrack_ep0020.pth
```

---

## Inference

### Setup dataset & metadata

Place the competition data under `competition_data/` and the two metadata files under `metadata/` as shown in the directory structure above.

### Run inference (public leaderboard)

```bash
python inference.py
```

This will:

1. Load the fine-tuned ORTrack model
2. Read all `public_lb` sequences from `metadata/contestant_manifest.json`
3. Track each sequence and collect per-frame bounding box predictions
4. Write results to `submission_output.csv` in the project root

### Output format

```
id,x,y,w,h
dataset2/basketball_player1_0,123,45,80,60
dataset2/basketball_player1_1,125,46,80,60
...
```

---

## Running on Hidden / Private Test Data

To run inference on the hidden test set, two small changes are needed in `inference.py`:

**1. Point `DATA_ROOT` to the folder containing the hidden sequences** (line 18):

```python
# Default (public data)
DATA_ROOT = os.path.join(BASE_DIR, 'competition_data')

# Change to your hidden data folder, for example:
DATA_ROOT = '/path/to/hidden/data'
```

**2. Change `SPLIT` to match the key used in your manifest** (line 19):

```python
# Default
SPLIT = 'public_lb'

# Change to whatever key your hidden manifest uses, for example:
SPLIT = 'hidden'
```

Then run inference exactly as normal:

```bash
python inference.py
```

> The hidden data must follow the same folder structure as the public data:
> each sequence is a folder containing a `.mp4` video file. The manifest must
> list the sequences under the chosen split key with `video_path` and `n_frames` fields.
> Annotations are not required — the tracker initializes from the first frame automatically.

---

## Scoring (Local Evaluation)

To evaluate predictions against annotated sequences locally:

```bash
python inference_score.py
```

---

## Efficiency Measurement

To profile FLOPs, parameters, latency, and model size:

```bash
python measure_efficiency.py
```

---

## Citation

If you use ORTrack, please cite the original paper:

```bibtex
@InProceedings{Wu_2025_CVPR,
  author    = {Wu, Yong and ...},
  title     = {Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking},
  booktitle = {CVPR},
  year      = {2025}
}
```
