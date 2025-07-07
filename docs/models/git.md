# GIT (Generative Image-to-Text) Model Training and Inference

## Supported Models

GIT (Generative Image-to-Text) models support various pre-trained checkpoints for image captioning.

## Training

### Basic Training

```bash
uv run python scripts/training/train_git.py \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints
```

### Training Configuration

Key training options:

- `--dataset`: Path to training data
- `--output_dir`: Where to save trained model
- `--model_id`: Base model to fine-tune
- `--epochs`: Training iterations (default: 5)
- `--batch_size`: Images per batch (default: 1)
- `--learning_rate`: Adaptation speed (default: 1e-4)
- `--target_modules`: Model layers to update
- `--rank`: LoRA adaptation intensity (default: 4)
- `--alpha`: LoRA weight scaling (default: 4)

### Advanced Training

```bash
uv run python scripts/training/train_git.py \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_checkpointing \
    --log_with wandb \
    --augment_images
```

### Unique GIT Training Features

- Image augmentation support
- Flexible block size configuration
- LoRA quantization (4, 8, or 16-bit)
- Advanced shuffling and dropout techniques

### Training Options

- `--augment_images`: Apply image transformations
- `--block_size`: Maximum sequence length
- `--lora_bits`: Quantization precision
- `--train_dir`: Alternative dataset specification

## Inference

### Basic Inference

```bash
uv run python scripts/inference/inference.py \
    --model_path ./checkpoints \
    --input_path /path/to/images \
    --output_path captions.jsonl
```

### Advanced Inference

```bash
uv run python scripts/inference/inference.py \
    --model_path ./checkpoints \
    --input_path image.jpg \
    --do_sample \
    --temperature 0.8 \
    --top_p 0.9
```

## Performance Tips

- Experiment with image augmentations
- Adjust block size for longer/shorter captions
- Use gradient checkpointing for large models
- Consider quantization for memory efficiency
