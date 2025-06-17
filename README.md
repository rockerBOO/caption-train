# Caption-Train

Train vision-language models for image captioning using 🤗 Transformers and PEFT (Parameter Efficient Fine-Tuning). This project provides a complete toolkit for training, evaluating, and deploying image captioning models.

<!--toc:start-->
- [Caption-Train](#caption-train)
  - [✨ Features](#✨-features)
  - [🚀 Quick Start](#🚀-quick-start)
  - [📦 Installation](#📦-installation)
  - [🎯 Supported Models](#🎯-supported-models)
  - [🛠️ Scripts Overview](#🛠️-scripts-overview)
  - [📊 Training](#📊-training)
  - [🔍 Inference](#🔍-inference)
  - [🧪 Development](#🧪-development)
  - [📚 Documentation](#📚-documentation)
<!--toc:end-->

## ✨ Features

- **🎨 Multiple Model Support**: Florence-2, BLIP, GIT models
- **⚡ Efficient Training**: LoRA/PEFT for memory-efficient fine-tuning
- **🔧 Modular Library**: Reusable components for custom workflows
- **📊 Comprehensive Evaluation**: Semantic similarity, CLIP scores, visual analysis
- **🌐 Production Ready**: Web APIs and deployment tools
- **📝 Rich Documentation**: Detailed guides and examples

## 🚀 Quick Start

### Train a Florence-2 Model
```bash
# Install dependencies
uv sync --extra cu124

# Train on your dataset
uv run python scripts/training/train_florence_refactored.py \
    --model_id microsoft/Florence-2-base-ft \
    --dataset /path/to/your/dataset \
    --output_dir ./checkpoints \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 1e-4
```

### Generate Captions
```bash
# Run inference on images
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path /path/to/images \
    --output_path captions.jsonl
```

### Start Web Server
```bash
# Deploy as web API
uv run python scripts/servers/server.py \
    --model_path ./checkpoints \
    --port 8000
```

## 📦 Installation

### Using UV (Recommended)
```bash
# Clone repository
git clone https://github.com/your-username/caption-train.git
cd caption-train

# Install with CUDA support
uv sync --extra cu124

# Or install with CPU support
uv sync --extra cpu
```

### Alternative Installation Methods
```bash
# Install development dependencies
uv sync --extra dev

# Install specific CUDA versions
uv sync --extra cu126  # CUDA 12.6
uv sync --extra cu121  # CUDA 12.1
```

## 🎯 Supported Models

| Model | Type | Best For | Memory |
|-------|------|----------|---------|
| **Florence-2** | Vision-Language | Detailed captions, multi-task | 8GB+ |
| **BLIP** | Image-to-Text | General captioning, fast inference | 6GB+ |
| **GIT** | Generative | Creative captions, long descriptions | 8GB+ |

### Model-Specific Features

- **Florence-2**: Task-specific prompts, multi-modal understanding
- **BLIP**: Bootstrapped training, robust performance
- **GIT**: Generative approach, flexible architecture

## 🛠️ Scripts Overview

### 🟢 Recommended Scripts (Latest)
All refactored scripts use the new library utilities for better maintainability:

| Script | Purpose | Models |
|--------|---------|---------|
| `train_florence_refactored.py` | Florence-2 training | Florence-2 |
| `train_blip_refactored.py` | BLIP model training | BLIP |
| `train_git_refactored.py` | GIT model training | GIT |
| `train2_refactored.py` | Auto-detect model type | All |
| `inference_refactored.py` | Universal inference | All |

### 🟡 Legacy Scripts
Original scripts maintained for backward compatibility:
- `train.py`, `train2.py`, `train3.py`
- `inference.py`, `inference_florence.py`

📁 **Detailed documentation available in each script directory:**
- [`scripts/training/README.md`](scripts/training/README.md) - Training guides and examples
- [`scripts/inference/README.md`](scripts/inference/README.md) - Inference and evaluation
- [`scripts/utils/README.md`](scripts/utils/README.md) - Dataset processing and analysis
- [`scripts/servers/README.md`](scripts/servers/README.md) - Web APIs and deployment

## 📊 Training

### Basic Training Example
```bash
# Florence-2 with custom configuration
uv run python scripts/training/train_florence_refactored.py \
    --config config.toml \
    --dataset /path/to/dataset \
    --output_dir ./models/florence-custom
```

### Configuration File (config.toml)
```toml
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
target_modules = ["qkv", "proj"]

[optimizer]
optimizer_name = "AdamW"
scheduler = "linear"
```

### Advanced Features
- **LoRA+**: Enhanced LoRA with differential learning rates
- **Gradient Compression**: Memory-efficient training with Flora
- **Mixed Precision**: Automatic FP16 optimization
- **Caption Augmentation**: Shuffle and dropout techniques

## 🔍 Inference

### Batch Processing
```bash
# Process entire directories
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path ./test_images/ \
    --output_path results.jsonl \
    --batch_size 16
```

### Custom Generation
```bash
# Creative sampling
uv run python scripts/inference/inference_refactored.py \
    --model_path ./checkpoints \
    --input_path image.jpg \
    --do_sample \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_length 256
```

## 🧪 Development

### Dependencies
- **[UV](https://docs.astral.sh/uv/)** for package management
- **[Ruff](https://docs.astral.sh/ruff/)** for linting and formatting
- **[Pytest](https://docs.pytest.org/)** for testing

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/caption_train/test_models_florence.py
uv run pytest tests/caption_train/test_training_pipeline.py

# Run with coverage
uv run pytest --cov=src/caption_train
```

### Code Quality
```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Type checking (if configured)
uv run mypy src/
```

### Current Test Status
✅ **157 tests passing** - All core functionality tested
- Model setup and configuration
- Training pipeline components
- Argument parsing and validation
- Utility functions and data processing
- Integration tests for refactored scripts

## 📚 Documentation

### Quick Links
- **[Training Guide](scripts/training/README.md)** - Comprehensive training documentation
- **[Inference Guide](scripts/inference/README.md)** - Inference and evaluation
- **[API Documentation](scripts/servers/README.md)** - Web server deployment
- **[Utilities Guide](scripts/utils/README.md)** - Dataset processing tools
- **[Library Documentation](docs/)** - Core library components

### Dataset Formats
The project supports multiple dataset formats:

**Directory Structure:**
```
dataset/
├── image1.jpg
├── image1.txt          # Caption for image1.jpg
├── image2.png
├── image2.txt
└── subdirectory/
    ├── image3.webp
    └── image3.txt
```

**JSONL Format:**
```jsonl
{"file_name": "image1.jpg", "text": "A red car on a street"}
{"file_name": "image2.png", "text": "A mountain landscape"}
```

**HuggingFace Datasets:**
```bash
--dataset "ybelkada/football-dataset"  # Remote dataset
```

### Performance Benchmarks

| Model | GPU | Batch Size | Images/min | Memory |
|-------|-----|------------|------------|---------|
| Florence-2 | RTX 4090 | 8 | ~150 | 12GB |
| BLIP-Base | RTX 3080 | 4 | ~120 | 8GB |
| GIT-Base | RTX 3080 | 4 | ~100 | 10GB |

### Community & Support

- **Issues**: [GitHub Issues](https://github.com/your-username/caption-train/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/caption-train/discussions)
- **Documentation**: Comprehensive guides in each script directory
- **Examples**: Real-world usage examples throughout documentation

### Contributing

We welcome contributions! Please see:
- Code follows Ruff formatting standards
- All tests pass: `uv run pytest`
- Documentation updated for new features
- Scripts include comprehensive help text

### License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

**⭐ Star this repo if you find it useful!**

*Caption-Train: Making vision-language model training accessible and efficient.*