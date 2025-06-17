# Installing Caption-Train with uv

uv is a modern Python package installer that can handle complex dependency requirements efficiently. Here's how to install caption-train with different extras using uv:

## Basic Installation

```bash
uv sync
```

## Installing with Specific CUDA Versions

```bash
# For CUDA 12.4 support
uv sync --extra cu124 --frozen

# For CUDA 12.1 support
uv sync --extra cu121 --frozen

# For CPU-only installation
uv sync --extra cpu --frozen
```

Note: The `--frozen` flag ensures exact versions are installed, preventing unexpected updates.

## Installing Feature-Specific Extras

```bash
# For server API components
uv sync --extra server --frozen

# For GPU monitoring
uv sync --extra gpu_metrics --frozen

# For ONNX runtime (limited support)
uv sync --extra onnx --frozen

# For LLM integration (accessing remote or local LLM servers0
uv sync --extra llm --frozen

# For Flash Attention
uv sync --extra flash-attn --frozen

# For Janus integration
uv sync --extra janus --frozen
```

## Installing Multiple Extras

You can combine multiple extras in a single command by repeating the `--extra` flag:

```bash
# CUDA 12.4 with server and GPU monitoring
uv sync --extra cu124 --extra server --extra gpu_metrics --frozen

# CPU version with ONNX and server support
uv sync --extra cpu --extra onnx --extra server --frozen
```

## Development Installation

```bash
# Install with development dependencies
uv sync --group dev --frozen
```

The package's pyproject.toml is configured with special uv settings that handle PyTorch's custom index URLs and prevent installation of conflicting CUDA versions automatically.
