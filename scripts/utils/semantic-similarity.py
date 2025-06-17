import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import statistics


def load_text(file_path):
    """Load text from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def calculate_similarity(model, text1, text2):
    """Calculate cosine similarity between two texts using sentence transformers."""
    # Encode texts to get embeddings
    embedding1 = model.encode([text1])[0]
    embedding2 = model.encode([text2])[0]

    # Reshape embeddings for cosine similarity calculation
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # Calculate cosine similarity and convert to Python float
    similarity = float(cosine_similarity(embedding1, embedding2)[0][0])
    return similarity


def find_image_text_pairs(directory):
    """Find all image-text pairs in the directory."""
    directory = Path(directory)

    # Get all jpg files
    image_files = list(directory.glob("*.jpg"))
    pairs = []

    for img_path in image_files:
        base_name = img_path.stem

        # Define the expected text file paths for this image
        txt_files = {
            "original": directory / f"{base_name}.txt",
            "generated": directory / f"{base_name}_generated.txt",
            "combined": directory / f"{base_name}_combined.txt",
            "improved": directory / f"{base_name}_combined_improved.txt",
        }

        # Check if all text files exist
        if all(txt_path.exists() for txt_path in txt_files.values()):
            pairs.append((base_name, txt_files))

    return pairs


# Helper function to make data JSON serializable
def make_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic similarity between image captions")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the image and text files")
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model to use (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="similarity_results.json",
        help="Output file to save results (default: similarity_results.json)",
    )
    args = parser.parse_args()

    dir_path = Path(args.dir)

    # Check if directory exists
    if not dir_path.is_dir():
        print(f"Error: {dir_path} is not a valid directory")
        return

    # Load sentence transformer model
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    # Find all image-text pairs
    pairs = find_image_text_pairs(dir_path)

    if not pairs:
        print("No complete image-text pairs found in the directory.")
        return

    print(f"Found {len(pairs)} complete image-text pairs.")

    # Prepare result data structure
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "directory": str(dir_path),
            "total_pairs": len(pairs),
        },
        "pair_results": {},
        "comparison_averages": {},
        "summary": {},
    }

    # Track all similarity scores by comparison type
    all_scores = {}

    # Process each pair
    for base_name, txt_files in pairs:
        print(f"\nProcessing image: {base_name}")

        # Load texts from files
        texts = {}
        for name, file_path in txt_files.items():
            texts[name] = load_text(file_path)

        # Calculate pairwise similarities
        pair_results = {}

        file_names = list(texts.keys())
        for i in range(len(file_names)):
            for j in range(i + 1, len(file_names)):
                name1 = file_names[i]
                name2 = file_names[j]

                comparison_key = f"{name1}_vs_{name2}"
                similarity = calculate_similarity(model, texts[name1], texts[name2])
                pair_results[comparison_key] = similarity

                # Track for averages
                if comparison_key not in all_scores:
                    all_scores[comparison_key] = []
                all_scores[comparison_key].append(similarity)

                print(f"  {comparison_key}: {similarity:.4f}")

        results["pair_results"][base_name] = pair_results

    # Calculate averages for each comparison type
    for comparison_key, scores in all_scores.items():
        results["comparison_averages"][comparison_key] = {
            "mean": float(statistics.mean(scores)),
            "median": float(statistics.median(scores)),
            "min": float(min(scores)),
            "max": float(max(scores)),
            "std_dev": float(statistics.stdev(scores) if len(scores) > 1 else 0),
            "count": len(scores),
        }

    # Generate overall summary
    best_comparison = max(results["comparison_averages"].items(), key=lambda x: x[1]["mean"])
    worst_comparison = min(results["comparison_averages"].items(), key=lambda x: x[1]["mean"])

    results["summary"] = {
        "total_comparisons": sum(len(scores) for scores in all_scores.values()),
        "best_performing_comparison": {"type": best_comparison[0], "mean_score": float(best_comparison[1]["mean"])},
        "worst_performing_comparison": {"type": worst_comparison[0], "mean_score": float(worst_comparison[1]["mean"])},
    }

    # Save results to file - ensure all values are JSON serializable
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=make_serializable)

    # Print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total image-text pairs processed: {len(pairs)}")
    print(f"Total comparison calculations: {results['summary']['total_comparisons']}")
    print("\nAverage similarity scores by comparison type:")
    for comp_type, stats in results["comparison_averages"].items():
        print(f"  {comp_type}: {stats['mean']:.4f} (± {stats['std_dev']:.4f})")

    print(
        f"\nBest performing comparison: {results['summary']['best_performing_comparison']['type']} "
        + f"with mean score of {results['summary']['best_performing_comparison']['mean_score']:.4f}"
    )
    print(
        f"Worst performing comparison: {results['summary']['worst_performing_comparison']['type']} "
        + f"with mean score of {results['summary']['worst_performing_comparison']['mean_score']:.4f}"
    )
    print("\nDetailed results saved to:", output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
