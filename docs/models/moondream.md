# Moondream Vision-Language Model

## Overview

Moondream is a compact Vision-Question-Answering (VQA) model integrated into the caption training pipeline.

## Captioning Usage

```bash
# Caption single image
uv run python scripts/models/moondream.py /path/to/image.png

# Caption multiple images
uv run python scripts/models/moondream.py \
    /path/to/image1.png \
    /path/to/image2.png

# Caption entire directory
uv run python scripts/models/moondream.py /path/to/images
```

## Configuration Options

- Positional arguments: Image files or directories
- `--device`: Specify processing device (cuda/cpu)
- `--epochs`: Model-specific parameter

### Processing Modes

- Single image processing
- Multiple image processing
- Directory-wide image processing

## Model Characteristics

- Lightweight VQA model
- FP16 support on CUDA
- Designed for efficient image understanding

## Dependencies

- Torch
- Bitsandbytes
- Transformers

## Performance Considerations

- Optimized for GPU (CUDA) processing
- Lower computational requirements
- Suitable for edge and mobile scenarios

## Recommended Workflow

1. Generate initial captions
2. Validate and refine with other models
3. Use for resource-constrained environments

## Limitations

- Smaller model size
- Less comprehensive than larger models
- May require additional fine-tuning

## Troubleshooting

- Verify CUDA and torch installation
- Check image format compatibility
- Use CPU mode if CUDA issues occur
