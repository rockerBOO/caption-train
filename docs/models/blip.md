# BLIP Model Training and Inference

## Supported Models

- Salesforce/blip-image-captioning-base
- Salesforce/blip-image-captioning-large

## Training

### Basic Training

```bash
uv run python scripts/training/train_blip.py \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --model_id Salesforce/blip-image-captioning-base
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
uv run python scripts/training/train_blip.py \
    --dataset /path/to/dataset \
    --output_dir ./checkpoints \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_checkpointing \
    --log_with wandb \
    --weight_decay 0.01
```

### Training Features

- Supports WandB, TensorBoard logging
- Gradient checkpointing for memory efficiency
- Caption shuffling and dropout
- Weight decay for regularization
- Flexible optimizer and scheduler options

## Inference

### Basic Inference

```bash
uv run python scripts/inference/inference.py \
    --model_path ./checkpoints \
    --input_path /path/to/images \
    --output_path captions.jsonl
```

### Inference Options

- `--model_path`: Path to trained model
- `--input_path`: Image or directory to caption
- `--output_path`: File to save generated captions
- `--batch_size`: Images processed per batch
- `--max_length`: Maximum caption length
- `--do_sample`: Enable creative sampling
- `--temperature`: Control randomness of outputs
- `--top_p`: Nucleus sampling parameter

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

- Adjust batch size based on GPU memory
- Use `do_sample` for more creative captions
- Experiment with temperature and top_p for diverse outputs
- Consider gradient checkpointing for large models
