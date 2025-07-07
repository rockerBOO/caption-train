# Dataset Management Utilities

## Find Missing Captions

Identify images in a dataset that lack corresponding text caption files.

### Basic Usage

```bash
uv run python scripts/find_missing_captions.py /path/to/dataset
```

### Configuration Options

- Positional argument: Path to dataset directory
- `--recursive` (`-r`): Recursively search subdirectories
- `--verbose` (`-v`): Provide detailed output about missing captions

### Advanced Usage

```bash
# Comprehensive dataset scan
uv run python scripts/find_missing_captions.py \
    /path/to/images \
    --recursive \
    --verbose
```

### Workflow

1. Scans the specified directory for image files
2. Checks for matching `.txt` caption files
3. Reports images without captions

### File Matching Rules

- Supports common image formats (jpg, png, webp, etc.)
- Expects caption files with same base filename
- Example matching:
  - `image1.jpg` → `image1.txt`
  - `image2.png` → `image2.txt`

### Use Cases

- Dataset validation
- Identify incomplete image-caption pairs
- Prepare datasets for training
- Ensure data completeness

### Best Practices

- Run before training to validate dataset
- Use recursive search for complex directory structures
- Combine with caption generation scripts
- Maintain consistent file naming conventions

## Extract Image Tags

Extract and process tags from JSONL files, creating caption files with extracted tags.

### Basic Usage

```bash
uv run python scripts/extract_tags.py input.jsonl
```

### Configuration Options

- `input_file`: Path to input JSONL file
- `--output-dir`: Directory to save caption files
- `--threshold`: Minimum confidence for tag inclusion
- `--separator`: Character to join multiple tags

### Advanced Usage

```bash
uv run python scripts/extract_tags.py \
    input.jsonl \
    --output-dir ./captions \
    --threshold 0.7 \
    --separator ", "
```

### Features

- Process JSONL files with image metadata
- Filter tags by confidence
- Customize tag output format
- Generate caption files from extracted tags

### Use Cases

- Convert tag-based annotations to captions
- Standardize image metadata
- Prepare datasets for training
- Tag-based image description generation

### Find Words in Captions

Search for specific words across text files in a directory.

### Basic Usage

```bash
uv run python scripts/find_words.py /path/to/directory
```

### Features

- Recursively search text files
- Case-insensitive word matching
- Support for custom file extensions
- Optional file output and deletion

### Advanced Usage

```bash
# Custom word and file extension
uv run python scripts/find_words.py \
    /path/to/captions \
    --word "specific_term" \
    --extension ".caption"
```

### Configuration

- Searches recursively through directories
- Matches whole words only
- Supports UTF-8 encoded text files

### Use Cases

- Content analysis
- Metadata extraction
- Text corpus exploration
- Filtering inappropriate content

## Image Feature Extraction with Florence

Extract advanced image features and generate captions using Florence models.

### Basic Usage

```bash
uv run python scripts/image_features_florence.py \
    --images /path/to/images
```

### Configuration Options

#### Model Selection

- `--base_model`: Florence-2 model variants
  - `microsoft/Florence-2-base-ft`
  - `microsoft/Florence-2-large-ft`
  - `microsoft/Florence-2-large`
  - `microsoft/Florence-2-base`
- `--peft_model`: Path to fine-tuned model

#### Processing Options

- `--task`: Captioning detail level
  - `<CAPTION>`: Basic description
  - `<DETAILED_CAPTION>`: More elaborate description
  - `<MORE_DETAILED_CAPTION>`: Comprehensive analysis
- `--batch_size`: Images processed per batch
- `--max_token_length`: Maximum caption length

### Advanced Usage

```bash
uv run python scripts/image_features_florence.py \
    --images /path/to/images \
    --base_model microsoft/Florence-2-large-ft \
    --task "<MORE_DETAILED_CAPTION>" \
    --batch_size 4 \
    --save_captions \
    --quantize
```

### Features

- Multi-model support
- Flexible captioning tasks
- Batch processing
- Optional caption saving
- 4-bit quantization support

### Use Cases

- Comprehensive image feature extraction
- Detailed image description generation
- Batch caption processing
- Model performance comparison

### Performance Considerations

- Choose appropriate model size
- Adjust batch size based on GPU memory
- Use quantization for reduced memory footprint

## Dataset Workflow

1. Find missing captions
2. Generate captions for missing images
3. Extract and analyze image features
4. Clean and prepare dataset

## Best Practices

- Regularly validate dataset completeness
- Use recursive search for complex directory structures
- Leverage verbose mode for detailed insights
