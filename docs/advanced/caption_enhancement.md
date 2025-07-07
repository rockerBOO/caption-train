# Advanced Caption Enhancement

## Overview

`enhance_captions.py` provides an advanced pipeline for improving image captions through Vision Language Models (VLM) and Large Language Models (LLM).

## Workflow

1. Load images and manual captions
2. Pre-generate captions using VLM
3. Cache VLM-generated captions
4. Combine and enhance captions using LLM
5. Cache final combined captions

## Basic Usage

```bash
uv run python scripts/enhance_captions.py \
  --dataset_dir images/ \
  --model_id microsoft/Florence-2-large-ft \
  --prompt "<MORE_DETAILED_CAPTION>" \
  --base_url http://127.0.0.1:5000/v1/
```

## Configuration Options

### VLM Configuration

- `--model_id`: Vision Language Model from Hugging Face
- `--prompt`: VLM caption generation prompt
- `--vlm_system_prompt`: VLM system prompt file
- `--revision`: Specific model revision
- `--trust_remote_code`: Trust remote model code

### LLM Configuration

- `--base_url`: LLM API endpoint (local server recommended)
- `--model`: Specific LLM model to use
- `--llm_system_prompt`: LLM system prompt file

### Dataset Options

- `--dataset_dir`: Directory with image/caption pairs
- `--recursive`: Search recursively for images
- `--combined_suffix`: Suffix for combined captions
- `--generated_suffix`: Suffix for VLM-generated captions

## Advanced Features

- Caching generated captions to disk
- Configurable data loading workers
- Supports local and remote LLM servers
- Flexible caption file naming

## Requirements

- Environment variable or .env file with OpenAI API key
- Transformers library
- Accelerate
- PIL
- tqdm

## Example Workflow

```bash
# Set OpenAI API key
export OPENAI_API_KEY=your_key_here

# Enhance captions using local LLM server
uv run python scripts/enhance_captions.py \
  --dataset_dir ./images \
  --model_id microsoft/Florence-2-large-ft \
  --base_url http://127.0.0.1:5000/v1/ \
  --prompt "<MORE_DETAILED_CAPTION>" \
  --cache_generated_captions_to_disk
```

## Performance Tips

- Use a local LLM server for faster processing
- Set appropriate `num_workers` for parallel processing
- Use `cache_generated_captions_to_disk` to resume interrupted jobs
