# Dataset Preparation

## Supported Dataset Formats

### Image/Text Pairs

```
dataset/
├── image1.jpg
├── image1.txt
├── image2.png
└── image2.txt
```

### JSONL Format

```jsonl
{"file_name": "image1.jpg", "text": "A red car on a street"}
{"file_name": "image2.png", "text": "A mountain landscape"}
```

## Preparing Your Dataset

Use `compile_captions.py` to convert datasets into a Hugging Face compatible format:

### Usage

```bash
# Basic conversion
uv run python scripts/utils/compile_captions.py /path/to/images output_dir

# Specify input and output directories
uv run python scripts/utils/compile_captions.py /path/to/images /path/to/output
```

### Key Features

- Converts Kohya-ss or image/text file pairs
- Generates `metadata.jsonl` compatible with Hugging Face Datasets
- Supports flexible input and output directory configuration

## Supported Sources

- Local image/text pairs
- Hugging Face datasets
- Kohya-ss style datasets

### Using Remote Datasets

```bash
# Train using a remote Hugging Face dataset
uv run python scripts/training/train_florence_peft.py \
    --dataset "ybelkada/football-dataset"
```

## Best Practices

- Ensure consistent file naming
- Use clear, descriptive captions
- Balance dataset size and diversity
- Preprocess images to consistent size/format
