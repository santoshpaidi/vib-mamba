# VIB-Mamba: Robust Vision-Based End-to-End Autonomous Driving

Official implementation of **VIB-Mamba**, a multi-modal state-space model for robust autonomous driving. This repository provides the core model architecture and the full two-stage pipeline for feature extraction and reproduction.

## Overview

VIB-Mamba addresses the kinematic instability and domain-overfitting common in attention-based end-to-end driving models. By combining a **Selective State-Space Model (Mamba)** with a **Variational Information Bottleneck (VIB)**, we achieve linear-time temporal reasoning and superior zero-shot robustness.

### Key Features
- **Linear-Time Inference:** Recursive state-space backbone for efficient long-context reasoning.
- **Information Hygiene:** VIB layer to prevent modality dominance and visual overfitting.
- **Two-Stage Pipeline:** Decoupled spatial feature extraction (frozen CLIP) and temporal modeling for high-throughput training.

## Repository Structure

```text
├── data/
│   ├── dataset.py               # Multi-modal dataset loaders
│   └── setup_protos.py          # Protocol Buffer setup for Waymo metrics
├── models/
│   ├── vib_layer.py             # Variational Information Bottleneck implementation
│   ├── vib_mamba_hf.py          # VIB-Mamba core architecture
│   └── vib_transformer.py       # VIB-Transformer baseline
├── scripts/
│   ├── extract_features.py      # Stage 1: Waymo CLIP feature extraction
│   ├── extract_nuscenes_aligned.py # Stage 1: nuScenes pre-alignment & extraction
│   ├── train.py                 # Stage 2: Training loop with VIB regularization
│   ├── evaluate.py              # Waymo kinematic evaluation
│   ├── evaluate_nuscenes.py     # nuScenes zero-shot behavioral analysis
│   └── inference.py             # Minimal inference wrapper
└── requirements.txt             # Dependency list
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Feature Extraction (Stage 1)
To extract CLIP-ViT-L/14 embeddings from Waymo TFRecords:
```bash
python scripts/extract_features.py --data_dir /path/to/waymo --output_dir ./embeddings
```

To pre-align and extract nuScenes for zero-shot evaluation:
```bash
python scripts/extract_nuscenes_aligned.py --nusc_path /path/to/nuscenes --output_path ./nusc_aligned
```

### 2. Training (Stage 2)
To train VIB-Mamba:
```bash
python scripts/train.py --data_dir ./embeddings --model_type mamba --beta 1e-4
```

### 3. Evaluation
To evaluate on nuScenes zero-shot:
```bash
python scripts/evaluate_nuscenes.py --data_dir ./nusc_aligned --checkpoints_dir ./checkpoints
```
