# Utility Scripts

This directory contains utility scripts for dataset processing, evaluation, and analysis. These tools help prepare datasets, evaluate model performance, and analyze training results.

## Dataset Processing

### `compile_captions.py`

**Purpose**: Convert individual caption files to metadata.jsonl format

```bash
# Convert directory of image/caption pairs to JSONL
uv run python scripts/utils/compile_captions.py \
    --dataset_dir /path/to/images_and_captions \
    --output_file metadata.jsonl

# Process with custom extensions
uv run python scripts/utils/compile_captions.py \
    --dataset_dir /path/to/dataset \
    --image_extensions .jpg .png .webp \
    --caption_extension .txt
```

**Input Format**:

```
dataset/
├── image1.jpg
├── image1.txt          # Caption for image1.jpg
├── image2.png
├── image2.txt          # Caption for image2.png
└── subdir/
    ├── image3.webp
    └── image3.txt      # Caption for image3.webp
```

**Output Format**:

```jsonl
{"file_name": "image1.jpg", "text": "A red car on a street"}
{"file_name": "image2.png", "text": "A mountain landscape"}
{"file_name": "subdir/image3.webp", "text": "A sunset over water"}
```

## Model Evaluation

### `semantic-similarity.py`

**Purpose**: Evaluate caption quality using semantic similarity metrics

```bash
# Compare generated captions with ground truth
uv run python scripts/utils/semantic-similarity.py \
    --predictions generated_captions.jsonl \
    --ground_truth test_captions.jsonl \
    --model_name sentence-transformers/all-MiniLM-L6-v2

# Batch evaluation with custom threshold
uv run python scripts/utils/semantic-similarity.py \
    --predictions captions.jsonl \
    --ground_truth ground_truth.jsonl \
    --threshold 0.8 \
    --output_file similarity_scores.json
```

**Input Format**:

```jsonl
# predictions.jsonl
{"file_name": "image1.jpg", "caption": "A red sports car"}
{"file_name": "image2.jpg", "caption": "Mountain vista"}

# ground_truth.jsonl  
{"file_name": "image1.jpg", "text": "A red car parked on street"}
{"file_name": "image2.jpg", "text": "A beautiful mountain landscape"}
```

**Output Metrics**:

- Mean semantic similarity score
- Per-image similarity scores
- Distribution statistics
- High/low similarity examples

### `clip-score-improvement.py`

**Purpose**: Measure CLIP score improvements after training

```bash
# Compare CLIP scores before/after training
uv run python scripts/utils/clip-score-improvement.py \
    --original_captions original.jsonl \
    --improved_captions improved.jsonl \
    --image_dir /path/to/images \
    --clip_model openai/clip-vit-base-patch32

# Detailed analysis with visualization
uv run python scripts/utils/clip-score-improvement.py \
    --original_captions baseline.jsonl \
    --improved_captions finetuned.jsonl \
    --image_dir ./test_images/ \
    --output_dir ./clip_analysis/ \
    --save_plots
```

**Metrics Computed**:

- CLIP similarity scores for original vs improved captions
- Score improvement distribution
- Statistical significance tests
- Per-category analysis (if categories provided)

## Image Analysis

### `swinv2.py`

**Purpose**: Extract image features using SwinV2 vision transformer

```bash
# Extract features for image similarity analysis
uv run python scripts/utils/swinv2.py \
    --image_dir /path/to/images \
    --output_file image_features.npy \
    --model_name microsoft/swinv2-base-patch4-window16-256

# Compute image similarity matrix
uv run python scripts/utils/swinv2.py \
    --image_dir ./dataset/images/ \
    --compute_similarity \
    --output_file similarity_matrix.json
```

**Use Cases**:

- Dataset diversity analysis
- Image clustering and deduplication
- Visual similarity search
- Quality assessment for training data

## Common Workflows

### Dataset Preparation Pipeline

```bash
# 1. Compile individual caption files
uv run python scripts/utils/compile_captions.py \
    --dataset_dir ./raw_dataset \
    --output_file ./processed/metadata.jsonl

# 2. Analyze dataset diversity
uv run python scripts/utils/swinv2.py \
    --image_dir ./raw_dataset \
    --compute_similarity \
    --output_file ./analysis/image_similarity.json

# 3. Generate baseline captions (if needed)
uv run python scripts/inference/inference_refactored.py \
    --model_path pretrained_model \
    --input_path ./raw_dataset \
    --output_path ./baselines/baseline_captions.jsonl
```

### Model Evaluation Pipeline

```bash
# 1. Generate captions with trained model
uv run python scripts/inference/inference_refactored.py \
    --model_path ./trained_model \
    --input_path ./test_images \
    --output_path ./results/generated_captions.jsonl

# 2. Evaluate semantic similarity
uv run python scripts/utils/semantic-similarity.py \
    --predictions ./results/generated_captions.jsonl \
    --ground_truth ./test_data/ground_truth.jsonl \
    --output_file ./results/semantic_scores.json

# 3. Evaluate CLIP score improvement
uv run python scripts/utils/clip-score-improvement.py \
    --original_captions ./baselines/baseline_captions.jsonl \
    --improved_captions ./results/generated_captions.jsonl \
    --image_dir ./test_images \
    --output_dir ./results/clip_analysis/
```

## Configuration Options

### Common Arguments

**Input/Output**:

- `--input_dir`, `--dataset_dir`: Input directory paths
- `--output_file`, `--output_dir`: Output file or directory
- `--image_extensions`: Supported image formats (default: .jpg, .png, .webp)

**Processing**:

- `--batch_size`: Batch size for processing (default: 32)
- `--num_workers`: Number of parallel workers (default: 4)
- `--device`: Device to use (cuda/cpu, auto-detected by default)

**Models**:

- `--model_name`: HuggingFace model identifier
- `--model_path`: Local model path

### Environment Variables

```bash
export HF_TOKEN="your-huggingface-token"  # For private models
export CUDA_VISIBLE_DEVICES="0"          # GPU selection
```

## Performance Tips

### Large Datasets

```bash
# Use larger batch sizes for faster processing
--batch_size 64 --num_workers 8

# Process in chunks for very large datasets
find /huge/dataset -name "*.jpg" | head -10000 | xargs -I {} python script.py
```

### Memory Management

```bash
# Reduce batch size if out of memory
--batch_size 16

# Use CPU for very large models on small GPUs
--device cpu
```

### Parallel Processing

```bash
# Multiple GPU processing
CUDA_VISIBLE_DEVICES=0 python script.py --batch_size 32 &
CUDA_VISIBLE_DEVICES=1 python script.py --batch_size 32 &
wait
```

## Output Formats

### JSONL (JSON Lines)

Most scripts output JSONL for easy streaming and processing:

```jsonl
{"file_name": "image1.jpg", "score": 0.85, "category": "nature"}
{"file_name": "image2.jpg", "score": 0.92, "category": "urban"}
```

### JSON Reports

Analysis scripts often output detailed JSON reports:

```json
{
    "summary": {
        "mean_score": 0.82,
        "std_score": 0.15,
        "num_samples": 1000
    },
    "per_category": {
        "nature": {"mean": 0.85, "count": 400},
        "urban": {"mean": 0.78, "count": 600}
    }
}
```

### CSV for Spreadsheet Analysis

```bash
# Convert JSONL to CSV for Excel/Google Sheets
python -c "
import pandas as pd
df = pd.read_json('results.jsonl', lines=True)
df.to_csv('results.csv', index=False)
"
```

## Troubleshooting

### Common Issues

**File Not Found Errors**:

- Check file paths are absolute or relative to current directory
- Verify input files exist and have correct permissions

**Memory Issues**:

- Reduce `--batch_size`
- Use `--device cpu` for large models
- Process datasets in smaller chunks

**Slow Processing**:

- Increase `--batch_size` (if memory allows)
- Use more `--num_workers`
- Ensure GPU is being utilized

### Getting Help

```bash
# Each script provides detailed help
uv run python scripts/utils/compile_captions.py --help
uv run python scripts/utils/semantic-similarity.py --help
```

## Integration with Training

These utilities integrate seamlessly with the training pipeline:

1. **Prepare datasets** with `compile_captions.py`
2. **Train models** with scripts from `../training/`
3. **Evaluate results** with `semantic-similarity.py` and `clip-score-improvement.py`
4. **Analyze improvements** with statistical and visual tools

This creates a complete machine learning workflow from data preparation to evaluation.
