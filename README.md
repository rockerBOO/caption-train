# Caption-Train

Train vision-language models for image captioning using efficient fine-tuning techniques.

## Quick Start

### Install

```bash
# Clone the repository
git clone https://github.com/your-username/caption-train.git
cd caption-train

# Install with CUDA 12.4 support
uv sync --extra cu124

# Or for CPU-only
uv sync --extra cpu
```

### Prepare Dataset

Convert image/text pairs to a compatible dataset:

```bash
uv run python scripts/utils/compile_captions.py /path/to/images output_dir
```

### Train a Model (Florence-2)

```bash
uv run accelerate launch scripts/training/train_florence_peft.py \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --epochs 5 \
    --batch_size 4
```

### Generate Captions

```bash
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path /path/to/images \
    --output_path captions.jsonl
```

## Supported Models

- Florence-2
- BLIP
- GIT (experimental)

## Documentation

- [Installation Guide](docs/installation.md)
- [Dataset Preparation](docs/datasets.md)
- [Model-specific Guides](docs/models/)
- [Advanced Techniques](docs/advanced/)

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint
uv run ruff check .
```

## License

MIT License
