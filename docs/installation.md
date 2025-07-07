# Installation Guide

## Prerequisites

- Python 3.10+
- CUDA (optional, for GPU support)

## Install with UV

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-username/caption-train.git
cd caption-train

# Install default dependencies
uv sync
```

### CUDA Support

Install with specific CUDA versions:

```bash
# CUDA 12.4 support
uv sync --extra cu124

# CUDA 12.1 support
uv sync --extra cu121

# CPU-only installation
uv sync --extra cpu
```

## Feature-Specific Extras

```bash
# Server API components
uv sync --extra server

# GPU monitoring
uv sync --extra gpu_metrics

# ONNX runtime
uv sync --extra onnx

# LLM integration
uv sync --extra llm

# Flash Attention
uv sync --extra flash-attn
```

### Multiple Extras

```bash
# Combine extras
uv sync --extra cu124 --extra server --extra gpu_metrics
```

## Development Installation

```bash
# Install with development dependencies
uv sync --group dev
```

## Troubleshooting

- Ensure compatible PyTorch version for your CUDA setup
- Check `pyproject.toml` for exact dependency requirements
