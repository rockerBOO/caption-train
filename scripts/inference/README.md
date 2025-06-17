# Inference Scripts

This directory contains scripts for running inference with trained vision-language models. These scripts can generate captions, evaluate model performance, and process images in batch.

## 🟢 Recommended Scripts

### `inference_refactored.py`

**Best for**: General-purpose inference with any trained model

```bash
# Single image inference
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path image.jpg \
    --output_path caption.txt

# Batch inference on directory
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path /path/to/images/ \
    --output_path captions.jsonl \
    --batch_size 8

# Custom generation parameters
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path /path/to/images/ \
    --do_sample \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_length 256 \
    --num_beams 3
```

## Generation Strategies

### Greedy Decoding (Default)

Fastest, most deterministic:

```bash
# No additional flags needed - this is the default
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path image.jpg
```

### Beam Search

Better quality, slower:

```bash
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path image.jpg \
    --num_beams 5
```

### Sampling

More creative, randomized:

```bash
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path image.jpg \
    --do_sample \
    --temperature 0.7 \
    --top_p 0.9
```

## Input Formats

### Single Image

```bash
--input_path /path/to/image.jpg
```

### Directory of Images

```bash
--input_path /path/to/images/
# Processes all images in the directory
```

### Specific Image Extensions

The script automatically processes common formats:

- `.jpg`, `.jpeg`
- `.png`
- `.webp`
- `.bmp`, `.tiff`

## Output Formats

### Single Caption File

```bash
--output_path caption.txt
# Writes: "A beautiful sunset over mountains"
```

### JSONL for Batch Processing

```bash
--output_path captions.jsonl
# Writes:
# {"file_name": "image1.jpg", "caption": "A red car on street"}
# {"file_name": "image2.png", "caption": "A mountain view"}
```

### Console Output

If no `--output_path` specified, captions are printed to console.

## Model Types

The inference script automatically detects and works with:

### Florence-2 Models

```bash
# Florence-2 specific prompts are automatically applied
uv run python scripts/inference/inference_refactored.py \
    --model_path ./florence_checkpoints \
    --input_path images/
```

### BLIP Models

```bash
# BLIP models work out of the box
uv run python scripts/inference/inference_refactored.py \
    --model_path ./blip_checkpoints \
    --input_path images/
```

### GIT Models

```bash
# GIT models with standard inference
uv run python scripts/inference/inference_refactored.py \
    --model_path ./git_checkpoints \
    --input_path images/
```

## Performance Optimization

### Batch Processing

```bash
--batch_size 8  # Process 8 images at once
# Increase for faster processing (limited by GPU memory)
```

### Memory Management

```bash
# For large images or limited memory
--batch_size 1
```

### GPU Utilization

```bash
# The script automatically uses GPU if available
# No additional configuration needed
```

## Advanced Usage

### Custom Prompts (Florence-2)

Florence-2 models support task-specific prompts:

```bash
# The script automatically applies appropriate prompts
# For Florence-2: "<MORE_DETAILED_CAPTION>" or "<DETAILED_CAPTION>"
```

### Long Captions

```bash
--max_length 512  # Allow longer captions (default: 256)
```

### Multiple Candidates

```bash
--num_beams 5  # Generate 5 candidate captions, return best
```

## Integration Examples

### Process Dataset for Evaluation

```bash
# Generate captions for test set
uv run python scripts/inference/inference_refactored.py \
    --model_path ./trained_model \
    --input_path ./test_images/ \
    --output_path test_captions.jsonl \
    --batch_size 16

# Use with evaluation scripts
uv run python scripts/utils/semantic-similarity.py \
    --predictions test_captions.jsonl \
    --ground_truth test_ground_truth.jsonl
```

### API Integration

```bash
# Generate captions for web service
uv run python scripts/inference/inference_refactored.py \
    --model_path ./production_model \
    --input_path /tmp/uploaded_image.jpg \
    --output_path /tmp/caption.txt \
    --temperature 0.7
```

## 🟡 Legacy Scripts

Maintained for backward compatibility:

### `inference.py`

Original inference script for GIT models:

```bash
uv run python scripts/inference/inference.py \
    --model_path ./checkpoints \
    --image_path image.jpg
```

### `inference_florence.py`

Florence-2 specific inference:

```bash
uv run python scripts/inference/inference_florence.py \
    --model_path ./florence_checkpoints \
    --image_dir ./images/
```

### `moondream.py`

Specialized script for Moondream models:

```bash
uv run python scripts/inference/moondream.py \
    --input_dir ./images/
```

**Migration**: Use `inference_refactored.py` for new projects as it supports all model types with a unified interface.

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:

```bash
--batch_size 1  # Reduce batch size
```

**Slow Inference**:

```bash
--batch_size 16  # Increase batch size (if memory allows)
```

**Poor Caption Quality**:

```bash
# Try different generation strategies
--num_beams 5  # Beam search
# or
--do_sample --temperature 0.8 --top_p 0.9  # Sampling
```

**No GPU Detected**:

- Ensure PyTorch CUDA is installed: `uv sync --extra cu124`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Getting Help

```bash
uv run python scripts/inference/inference_refactored.py --help
```

## Performance Benchmarks

Typical performance on different hardware:

### RTX 4090 (24GB)

- **Batch size 16**: ~200 images/minute
- **Batch size 8**: ~150 images/minute
- **Single image**: ~50 images/minute

### RTX 3080 (10GB)

- **Batch size 8**: ~120 images/minute
- **Batch size 4**: ~90 images/minute
- **Single image**: ~35 images/minute

### CPU Only

- **Single image**: ~5-10 images/minute
- Use for small-scale processing or testing

_Performance varies by model size, image resolution, and generation parameters._
