# Caption Improvement Techniques

## Overview

The project provides multiple scripts for enhancing image captions through advanced AI techniques:

- `enhance_captions.py`
- `evaluate-and-improve.py`

## Workflow

1. Load images and manual captions
2. Pre-generate captions using Vision Language Model (VLM)
3. Cache VLM-generated captions
4. Combine and refine captions using Large Language Model (LLM)
5. Cache final enhanced captions

## Basic Usage

```bash
# Enhance captions in a directory
uv run python evaluate-and-improve.py \
    --dataset_dir /path/to/images \
    --model_id microsoft/Florence-2-large-ft \
    --base_url http://127.0.0.1:5000/v1/ \
    --prompt "<MORE_DETAILED_CAPTION>"
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

## Example Workflow

```bash
# Set OpenAI API key
export OPENAI_API_KEY=your_key_here

# Enhance captions using local LLM server
uv run python evaluate-and-improve.py \
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

## Caption Enhancement Strategies

1. **VLM Pre-generation**
   - Generate initial captions with Vision Language Models
   - Capture basic image understanding

2. **LLM Refinement**
   - Use Large Language Models to improve captions
   - Add context, detail, and creativity
   - Standardize caption style and quality

## Requirements

- Environment variable or .env file with OpenAI API key
- Transformers library
- Accelerate
- PIL
- tqdm

## Recommended Use Cases

- Improve dataset caption quality
- Standardize caption descriptions
- Generate more detailed, contextual captions
- Prepare datasets for advanced training
