# Qwen 2.5 Vision-Language Model

## Overview

Qwen 2.5 VL is an advanced multimodal AI model that can analyze and describe images with high precision.

## Key Features

- 3B parameter Vision-Language model
- Flash Attention 2 acceleration
- Supports image and text inputs
- Advanced image description generation

## Model Configuration

- Model: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`
- Precision: Float16
- Attention: Flash Attention 2
- Device Mapping: Automatic

## Usage Patterns

### Image Description

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_file},
            {"type": "text", "text": "Describe this image"}
        ]
    }
]
```

### Performance Optimization

- Configurable visual token range
- Supports min/max pixel configurations
- Automatic device placement

## Pixel Token Configuration

```python
# Example custom pixel token range
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
```

## Dependencies

- transformers
- torch
- qwen-vl-utils
- accelerate
- PIL

## Recommended Workflow

1. Prepare image and text prompt
2. Process with Qwen 2.5 VL processor
3. Generate detailed description
4. Post-process or refine output

## Performance Considerations

- Uses AWQ (Activation-aware Weight Quantization)
- Optimized for GPU inference
- Balance token range for performance

## Limitations

- Computational intensity for large images
- Requires compatible GPU
- Performance varies with image complexity

## Troubleshooting

- Ensure CUDA compatibility
- Check GPU memory requirements
- Adjust token range if processing fails
- Verify image format support
