# Vision-Language Models Documentation

## Supported Models

### BLIP

- [BLIP Model Guide](blip.md)
  - Training and fine-tuning configurations
  - Inference strategies
  - Performance optimization

### Florence

- [Florence-2 Model Guide](florence.md)
  - Advanced captioning with Florence-2
  - Model variants and use cases
  - Training and inference details

### GIT (Generative Image-to-Text)

- [GIT Model Guide](git.md)
  - Generative image captioning
  - Training configurations
  - Inference techniques

### Experimental Models

#### Moondream

- [Moondream Model](moondream.md)
  - Compact vision-language model
  - Experimental support
  - Lightweight captioning

#### Qwen

- [Qwen 2.5 VL](qwen.md)
  - Advanced multimodal AI model
  - Image description generation
  - Performance considerations

## Model Comparison

| Model       | Parameters   | Best For                  | Memory | Complexity |
| ----------- | ------------ | ------------------------- | ------ | ---------- |
| BLIP        | Small-Medium | General Captioning        | 6GB+   | Low        |
| Florence-2  | Medium-Large | Detailed Captions         | 8GB+   | Medium     |
| GIT         | Large        | Creative Descriptions     | 10GB+  | High       |
| Moondream   | Small        | Edge/Mobile               | 2-4GB  | Low        |
| Qwen 2.5 VL | Medium       | Multi-modal Understanding | 6-8GB  | Medium     |

## Recommended Workflow

1. Understand model capabilities
2. Choose appropriate model for your use case
3. Fine-tune with your dataset
4. Evaluate and iterate

## Contributing

- Share model performance insights
- Contribute model-specific improvements
- Add new model support
