# Caption-Train Scripts

This directory contains standalone scripts for training, inference, utilities, and server functionality. All scripts use the refactored caption-train library for consistent, maintainable code.

## Directory Structure

```
scripts/
├── training/           # Model training scripts
├── inference/          # Inference and evaluation scripts
├── utils/              # Dataset processing and analysis utilities
├── servers/            # Web servers and API endpoints
└── README.md          # This file
```

## Quick Start

### Training a Model

```bash
# Florence-2 model training
uv run python scripts/training/train_florence_refactored.py \
    --model_id microsoft/Florence-2-base-ft \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --epochs 5 \
    --batch_size 4

# Multi-model training (auto-detects model type)
uv run python scripts/training/train2_refactored.py \
    --model_name_or_path microsoft/git-base \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints
```

### Running Inference

```bash
# General inference script
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path /path/to/images \
    --output_path captions.jsonl
```

### Processing Datasets

```bash
# Compile individual caption files into metadata.jsonl
uv run python scripts/utils/compile_captions.py \
    --dataset_dir /path/to/images_and_captions
```

## Script Categories

### 🟢 Recommended Scripts (Refactored)

These scripts use the new library utilities and are actively maintained:

- **Training**: `train_*_refactored.py` scripts
- **Inference**: `inference_refactored.py`
- **All utility and server scripts** (already using best practices)

### 🟡 Legacy Scripts

Original scripts maintained for backward compatibility:

- `train.py`, `train2.py`, `train3.py`, etc.
- `inference.py`, `inference_florence.py`

**Note**: Legacy scripts will continue to work but new features and improvements are added to the refactored versions.

## Configuration Options

All refactored scripts support:

- **TOML configuration files**: `--config config.toml`
- **Command-line arguments**: Override any config file setting
- **Environment variables**: For API keys and paths
- **Flexible model types**: Auto-detection and manual specification

## Getting Help

Each script provides detailed help:

```bash
uv run python scripts/training/train_florence_refactored.py --help
```

For comprehensive documentation, see the individual README files in each subdirectory.
