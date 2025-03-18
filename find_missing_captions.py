import argparse
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify missing .txt files paired with image files in a dataset directory."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to the dataset directory containing image files."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search recursively through subdirectories."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Increase output verbosity."
    )
    return parser.parse_args()

def find_missing_txt_pairs(args):
    dataset_dir = args.dataset_dir
    recursive = args.recursive
    verbose = args.verbose

    if not dataset_dir.exists():
        print(f"Error: The directory '{dataset_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    images = []
    if recursive:
        images = (
            list(dataset_dir.rglob("*.png"))
            + list(dataset_dir.rglob("*.jpg"))
            + list(dataset_dir.rglob("*.jpeg"))
            + list(dataset_dir.rglob("*.webp"))
        )
    else:
        images = (
            list(dataset_dir.glob("*.png"))
            + list(dataset_dir.glob("*.jpg"))
            + list(dataset_dir.glob("*.jpeg"))
            + list(dataset_dir.glob("*.webp"))
        )

    missing_pairs = []
    for image_path in images:
        base_name = image_path.stem
        txt_path = image_path.parent / f"{base_name}.txt"
        if not txt_path.exists():
            missing_pairs.append(image_path)
            if verbose:
                print(f"Missing .txt file for: {image_path}", file=sys.stderr)

    return missing_pairs

def main(args):
    missing = find_missing_txt_pairs(args)
    
    if missing:
        print("Missing .txt pairs:")
        for path in missing:
            print(path)
    else:
        print("All image files have corresponding .txt files.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
