import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate CLIP scores for image-caption pairs")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing image-caption pairs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)",
    )
    return parser.parse_args()


def load_image_caption_pairs(directory):
    """Load image and corresponding caption pairs from directory."""
    directory = Path(directory)
    image_caption_pairs = []

    for image_path in directory.glob("*.jpg"):
        original_path = directory / f"{image_path.stem}.txt"
        generated_path = directory / f"{image_path.stem}_generated.txt"
        combined_path = directory / f"{image_path.stem}_combined.txt"
        improved_path = directory / f"{image_path.stem}_combined_improved.txt"

        # Check if all files exist
        files_exist = all(p.exists() for p in [original_path, generated_path, combined_path, improved_path])

        if files_exist:
            captions = {}
            for name, path in [
                ("original", original_path),
                ("generated", generated_path),
                ("combined", combined_path),
                ("improved", improved_path),
            ]:
                with open(path, "r", encoding="utf-8") as f:
                    captions[name] = f.read().strip()

            image_caption_pairs.append(
                {
                    "image_path": image_path,
                    "original_caption": captions["original"],
                    "generated_caption": captions["generated"],
                    "combined_caption": captions["combined"],
                    "improved_caption": captions["improved"],
                }
            )

    return image_caption_pairs


def calculate_clip_scores(model, processor, pairs, batch_size, device):
    """Calculate CLIP scores for image-caption pairs in batches."""
    results = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="Processing batches"):
        batch_pairs = pairs[i : i + batch_size]
        images = [Image.open(pair["image_path"]).convert("RGB") for pair in batch_pairs]

        caption_types = ["original", "generated", "combined", "improved"]
        scores = {}

        # Process each caption type
        for caption_type in caption_types:
            captions = [pair[f"{caption_type}_caption"] for pair in batch_pairs]

            inputs = processor(
                text=captions, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                scores[caption_type] = outputs.logits_per_image.diagonal().cpu().numpy()

        for j, pair in enumerate(batch_pairs):
            result = {
                "image_path": pair["image_path"].name,
            }

            # Add all captions
            for caption_type in caption_types:
                result[f"{caption_type}_caption"] = pair[f"{caption_type}_caption"]
                result[f"{caption_type}_score"] = scores[caption_type][j]

            # Calculate improvements between stages
            result["original_to_generated_diff"] = scores["generated"][j] - scores["original"][j]
            result["generated_to_combined_diff"] = scores["combined"][j] - scores["generated"][j]
            result["combined_to_improved_diff"] = scores["improved"][j] - scores["combined"][j]
            result["overall_improvement"] = scores["improved"][j] - scores["original"][j]

            results.append(result)

    return results


def main(args):
    print(f"Loading image-caption pairs from {args.directory}")
    pairs = load_image_caption_pairs(args.directory)
    print(f"Found {len(pairs)} complete image-caption pairs")

    if not pairs:
        print("No valid image-caption pairs found. Exiting.")
        return

    print(f"Loading CLIP model on {args.device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"Calculating CLIP scores with batch size {args.batch_size}")
    results = calculate_clip_scores(model, processor, pairs, args.batch_size, args.device)

    # Calculate statistics for each stage comparison
    comparisons = [
        ("original", "generated", "original_to_generated_diff"),
        ("generated", "combined", "generated_to_combined_diff"),
        ("combined", "improved", "combined_to_improved_diff"),
        ("original", "improved", "overall_improvement"),
    ]

    # Save detailed results
    output_dir = Path(args.directory) / "clip_results"
    output_dir.mkdir(exist_ok=True)

    # Save summary
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        for before, after, diff_key in comparisons:
            scores_before = [r[f"{before}_score"] for r in results]
            scores_after = [r[f"{after}_score"] for r in results]
            differences = [r[diff_key] for r in results]

            f.write(f"\n{before.capitalize()} → {after.capitalize()} Comparison:\n")
            f.write(f"Average {before} score: {np.mean(scores_before):.4f}\n")
            f.write(f"Average {after} score: {np.mean(scores_after):.4f}\n")
            f.write(f"Average improvement: {np.mean(differences):.4f}\n")
            f.write(f"Number of improved captions: {sum(d > 0 for d in differences)} out of {len(differences)}\n")

            print(f"\n{before.capitalize()} → {after.capitalize()} Comparison:")
            print(f"Average {before} score: {np.mean(scores_before):.4f}")
            print(f"Average {after} score: {np.mean(scores_after):.4f}")
            print(f"Average improvement: {np.mean(differences):.4f}")
            print(f"Number of improved captions: {sum(d > 0 for d in differences)} out of {len(differences)}")

    # Save detailed results
    results_sorted = sorted(results, key=lambda x: x["overall_improvement"], reverse=True)
    with open(output_dir / "detailed_results.txt", "w", encoding="utf-8") as f:
        for r in results_sorted:
            f.write(f"Image: {r['image_path']}\n\n")

            # Original caption
            f.write(f"1. Original caption: {r['original_caption']}\n")
            f.write(f"   Score: {r['original_score']:.4f}\n\n")

            # Generated caption
            f.write(f"2. Generated caption: {r['generated_caption']}\n")
            f.write(f"   Score: {r['generated_score']:.4f}\n")
            f.write(f"   Improvement from original: {r['original_to_generated_diff']:.4f}\n\n")

            # Combined caption
            f.write(f"3. Combined caption: {r['combined_caption']}\n")
            f.write(f"   Score: {r['combined_score']:.4f}\n")
            f.write(f"   Improvement from generated: {r['generated_to_combined_diff']:.4f}\n\n")

            # Improved caption
            f.write(f"4. Improved caption: {r['improved_caption']}\n")
            f.write(f"   Score: {r['improved_score']:.4f}\n")
            f.write(f"   Improvement from combined: {r['combined_to_improved_diff']:.4f}\n\n")

            # Overall improvement
            f.write(f"Overall improvement: {r['overall_improvement']:.4f}\n")
            f.write("\n---\n\n")

    # Save CSV for easy analysis
    with open(output_dir / "results.csv", "w", encoding="utf-8") as f:
        # Write header
        headers = [
            "image",
            "original_score",
            "generated_score",
            "combined_score",
            "improved_score",
            "original_to_generated",
            "generated_to_combined",
            "combined_to_improved",
            "overall_improvement",
        ]
        f.write(",".join(headers) + "\n")

        # Write data
        for r in results:
            row = [
                r["image_path"],
                f"{r['original_score']:.4f}",
                f"{r['generated_score']:.4f}",
                f"{r['combined_score']:.4f}",
                f"{r['improved_score']:.4f}",
                f"{r['original_to_generated_diff']:.4f}",
                f"{r['generated_to_combined_diff']:.4f}",
                f"{r['combined_to_improved_diff']:.4f}",
                f"{r['overall_improvement']:.4f}",
            ]
            f.write(",".join(row) + "\n")

    print(f"Detailed results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
