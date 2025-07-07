# Caption Training Utilities

## Semantic Similarity Analysis

Evaluate the semantic similarity between generated and original captions using sentence transformers.

### Basic Usage

```bash
uv run python scripts/utils/semantic-similarity.py \
    --dir /path/to/captions
```

### Configuration Options

- `--dir`: Directory containing image and text files
- `--model`: Sentence transformer model
  - Default: `all-MiniLM-L6-v2`
  - Supports various sentence transformer models
- `--output`: Output file for similarity results
  - Default: `similarity_results.json`

### Advanced Usage

```bash
uv run python scripts/utils/semantic-similarity.py \
    --dir ./captions \
    --model sentence-transformers/all-mpnet-base-v2 \
    --output custom_similarity.json
```

### Recommended Models

- `all-MiniLM-L6-v2`: Fast, lightweight
- `all-mpnet-base-v2`: Higher accuracy
- `multi-qa-MiniLM-L6-dot-v1`: Good for question-answering tasks

### Use Cases

- Evaluate caption quality and consistency
- Compare different captioning models
- Assess fine-tuning effectiveness
- Measure semantic drift in generated captions

### Interpretation

- Similarity scores range from 0 to 1
- Higher scores indicate more semantic alignment
- Helps quantify caption quality beyond exact text matching

### Performance Considerations

- Choose model based on computational resources
- Larger models provide more nuanced similarity assessment
- Consider batch processing for large datasets

## Other Utility Scripts

### Caption Compilation

```bash
# Convert image/text pairs to Hugging Face dataset
uv run python scripts/utils/compile_captions.py \
    /path/to/images /path/to/output
```

### CLIP Score Analysis

Calculate and analyze CLIP scores for image-caption pairs to evaluate caption quality.

```bash
# Analyze CLIP scores for a directory of images and captions
uv run python scripts/utils/clip-score-improvement.py \
    --directory /path/to/image-caption-pairs \
    --batch_size 32 \
    --device cuda
```

#### Configuration Options

- `--directory`: Path to directory with image-caption pairs
- `--batch_size`: Number of pairs processed simultaneously
- `--device`: Processing device (cuda or cpu)

#### Use Cases

- Evaluate caption quality
- Compare different captioning models
- Identify captions that need improvement

#### Understanding CLIP Scores

- Higher scores indicate better semantic alignment
- Ranges typically between 0 and 1
- Helps quantify image-text relevance

## Performance Considerations

- Choose appropriate transformer models
- Consider computational resources
- Use GPU for faster processing

## Recommended Workflow

1. Generate captions
2. Run semantic similarity analysis
3. Use insights to refine captioning models
