# Advanced Training Techniques

## External Dataset Training (train3.py)

Train vision-language models on external Hugging Face datasets with advanced configuration options.

### Basic Usage

```bash
uv run python scripts/training/train3.py \
    --model_id Salesforce/blip-image-captioning-base \
    --dataset_name ybelkada/football-dataset \
    --output_dir ./football_captions
```

### Configuration Options

#### Model Selection

- `--model_id`: Hugging Face model identifier
- `--dataset_name`: External dataset to train on
- `--split`: Specific dataset split (train/validation)

#### Training Hyperparameters

- `--learning_rate`: Adaptation speed (default: 1e-4)
- `--weight_decay`: Regularization strength
- `--batch_size`: Images processed per iteration
- `--epochs`: Total training iterations
- `--sample_every_n_steps`: Periodic model output sampling

#### LoRA Configuration

- `--rank`: LoRA adaptation intensity
- `--alpha`: LoRA weight scaling
- `--dropout`: Regularization dropout rate

#### Environment

- `--device`: Training hardware (cuda/cpu)
- `--seed`: Reproducibility control

### Advanced Example

```bash
uv run python scripts/training/train3.py \
    --model_id Salesforce/blip-image-captioning-base \
    --dataset_name ybelkada/football-dataset \
    --output_dir ./football_captions \
    --learning_rate 2e-4 \
    --batch_size 8 \
    --epochs 10 \
    --rank 16 \
    --alpha 32 \
    --device cuda
```

## Training Strategies

### External Dataset Benefits

- Access to diverse, pre-curated datasets
- Quick experimentation
- Reduced data preparation overhead

### Performance Optimization

- Adjust batch size based on GPU memory
- Use gradient accumulation for larger effective batch sizes
- Experiment with learning rates
- Implement early stopping

### Monitoring

- Track validation loss
- Use TensorBoard or WandB for visualization
- Log key metrics periodically

## Best Practices

- Start with pre-trained models
- Use smaller learning rates for fine-tuning
- Implement learning rate scheduling
- Use weight decay for regularization
- Monitor for overfitting

## Troubleshooting

- Insufficient GPU memory: Reduce batch size
- Poor performance: Adjust learning rate
- Dataset issues: Verify dataset compatibility
- Reproducibility: Set consistent random seed
