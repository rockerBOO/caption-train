import json
from pathlib import Path


def process_jsonl(input_file, output_dir=None, threshold=0.2, separator=", "):
    """
    Process a JSONL file to extract tags above the threshold and create caption files.

    Args:
        input_file: Path to the input JSONL file
        output_dir: Directory to save the caption files (defaults to same as input)
        threshold: Minimum confidence score for tags to include
        separator: String to use when joining tags
    """
    input_path = Path(input_file)

    # If no output directory is specified, use the input file's directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    with input_path.open("r") as f:
        for line in f:
            data = json.loads(line.strip())
            filename = data["filename"]
            tags = data["tags"]

            # Filter tags by threshold and remove underscores
            filtered_tags = []
            for tag, confidence in tags.items():
                # Skip tags that start with "rating:"
                if tag.startswith("rating:"):
                    continue

                if confidence >= threshold:
                    # Replace underscores with spaces
                    clean_tag = tag.replace("_", " ")
                    filtered_tags.append(clean_tag)

            # Create caption text
            caption = separator.join(filtered_tags)

            # Generate output filename (replacing original extension with .txt)
            base_name = Path(filename).stem
            output_file = output_dir / f"{base_name}.txt"

            # Write caption to file
            output_file.write_text(caption)

            print(f"Created caption file for {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract tags from JSONL and create caption files")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("--output-dir", help="Directory to save caption files")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum confidence threshold for tags")
    parser.add_argument("--separator", default=", ", help="Separator to use between tags")

    args = parser.parse_args()

    process_jsonl(args.input_file, args.output_dir, args.threshold, args.separator)
