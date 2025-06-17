# Training Scripts

This directory contains scripts for training vision-language models using LoRA (Low-Rank Adaptation) fine-tuning. All scripts use the caption-train library for consistent, maintainable training pipelines.

## 🚀 Available Training Scripts

### `train_florence.py`
**Best for**: Florence-2 model fine-tuning
```bash
uv run python scripts/training/train_florence.py \
    --model_id microsoft/Florence-2-base-ft \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --rank 8 \
    --alpha 16 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --epochs 5 \
    --prompt "<MORE_DETAILED_CAPTION>"
```

### `train_git.py`
**Best for**: GIT model fine-tuning
```bash
uv run python scripts/training/train_git.py \
    --model_id microsoft/git-base \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --augment_images \
    --block_size 1024 \
    --lora_bits 4
```

### `train_blip.py`
**Best for**: BLIP model fine-tuning
```bash
uv run python scripts/training/train_blip.py \
    --model_id Salesforce/blip-image-captioning-base \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --weight_decay 1e-4 \
    --learning_rate 2e-5
```

### `train3.py`
**Best for**: Multi-model training with auto-detection
```bash
# Auto-detects model type from model name
uv run python scripts/training/train3.py \
    --model_name_or_path microsoft/Florence-2-base-ft \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --epochs 3 \
    --batch_size 2
```

### `fine_tune_blip_using_peft.py`
**Best for**: Advanced BLIP fine-tuning with custom PEFT configurations
```bash
uv run python scripts/training/fine_tune_blip_using_peft.py \
    --model_id Salesforce/blip-image-captioning-large \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --rank 16 \
    --alpha 32
```

## Configuration Options

### Command Line Arguments
All scripts support comprehensive argument configuration:
- **Model**: `--model_id`, `--model_name_or_path`
- **Dataset**: `--dataset`, `--dataset_dir` (directory with image/caption pairs)
- **Training**: `--epochs`, `--batch_size`, `--learning_rate`, `--gradient_accumulation_steps`
- **LoRA**: `--rank`, `--alpha`, `--target_modules`, `--rslora`
- **Optimization**: `--optimizer_name`, `--scheduler`, `--weight_decay`
- **Logging**: `--log_with wandb`, `--name experiment_name`

### TOML Configuration Files
Create reusable configurations:
```toml
# config.toml
[model]
model_id = "microsoft/Florence-2-base-ft"

[training]
epochs = 5
batch_size = 4
learning_rate = 1e-4
gradient_accumulation_steps = 2

[peft]
rank = 8
alpha = 16
rslora = false

[optimizer]
optimizer_name = "AdamW"
scheduler = "linear"

[dataset]
dataset_dir = "/path/to/dataset"
recursive = true
num_workers = 4
```

Use with: `--config config.toml`

### Environment Variables
```bash
export WANDB_API_KEY="your-wandb-key"
export HF_TOKEN="your-huggingface-token"
```

## Dataset Formats

### Directory-based Dataset
```
dataset/
├── image1.jpg
├── image1.txt          # Caption for image1.jpg
├── image2.png
├── image2.txt          # Caption for image2.png
└── subdir/
    ├── image3.webp
    └── image3.txt      # Caption for image3.webp
```

### JSONL Dataset
```jsonl
{"file_name": "image1.jpg", "text": "A red car on a street"}
{"file_name": "image2.png", "text": "A mountain landscape"}
```

### HuggingFace Dataset
```bash
--dataset "ybelkada/football-dataset"  # HF dataset name
```

## Advanced Features

### LoRA+ Training
Enhanced LoRA with differential learning rates:
```bash
--lora_plus_ratio 16  # B matrices get 16x learning rate
```

### Gradient Compression
Memory-efficient training with Flora:
```bash
--accumulation_rank 8  # Low-rank gradient compression
```

### Mixed Precision & Quantization
```bash
--quantize              # 4-bit model quantization
--gradient_checkpointing # Reduce memory usage
```

### Caption Augmentation
```bash
--shuffle_captions      # Randomly shuffle caption parts
--caption_dropout 0.1   # Randomly drop 10% of caption parts
--frozen_parts 1        # Keep first part fixed when shuffling
```

## Model-Specific Notes

### Florence-2
- **Best prompts**: `"<MORE_DETAILED_CAPTION>"`, `"<DETAILED_CAPTION>"`
- **Target modules**: `["qkv", "proj", "fc1", "fc2"]` (auto-configured)
- **Typical LR**: `1e-4` to `1e-5`

### GIT
- **Image augmentation**: `--augment_images` recommended
- **Sequence length**: `--block_size 2048` (default) or `1024` for memory constraints
- **Quantization**: `--lora_bits 4` for memory efficiency

### BLIP
- **Weight decay**: `1e-4` recommended for stability
- **Conservative LR**: `2e-5` to `5e-5` (lower than other models)
- **Batch size**: Start with 2-4, BLIP models are memory-intensive

## Script Features

All training scripts include:
- **LoRA/PEFT Support**: Memory-efficient fine-tuning
- **Mixed Precision**: Automatic FP16/BF16 optimization  
- **Gradient Checkpointing**: Reduce memory usage
- **Wandb Integration**: Experiment tracking and logging
- **TOML Configuration**: Reusable configuration files
- **Multi-GPU Support**: Distributed training with Accelerate

## Troubleshooting

### Common Issues

**Out of Memory**:
```bash
--batch_size 1 --gradient_accumulation_steps 4 --quantize --gradient_checkpointing
```

**Slow Training**:
```bash
--num_workers 4 --accumulation_rank 8  # Parallel data loading + compression
```

**Poor Convergence**:
```bash
--learning_rate 5e-5 --scheduler linear --lora_plus_ratio 16
```

### Getting Help
```bash
uv run python scripts/training/train_florence_refactored.py --help
```

## Performance Tips

1. **Start small**: Begin with `--epochs 1 --batch_size 1` to verify setup
2. **Monitor memory**: Use `nvidia-smi` to track GPU utilization
3. **Use wandb**: `--log_with wandb --name my_experiment` for tracking
4. **Gradient accumulation**: Simulate larger batches with `--gradient_accumulation_steps`
5. **Mixed precision**: Automatically enabled by Accelerate for compatible GPUs