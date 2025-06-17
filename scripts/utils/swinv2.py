import numpy as np
import argparse
from PIL import Image, ImageOps
from pathlib import Path
import onnxruntime as ort
import torchvision.transforms as transforms
import csv


def load_and_preprocess_image(image_input, target_size=(448, 448), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Load an image from a file path, numpy array, or PIL.Image,
    then preprocess it to a fixed target size using letterbox padding.
    Returns a numpy array with shape (1, H, W, C) in NHWC format.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type. Expected file path, numpy array, or PIL.Image.")

    iw, ih = image.size
    tw, th = target_size

    # Calculate scaling factor preserving aspect ratio.
    scale = min(tw / iw, th / ih)
    new_size = (int(iw * scale), int(ih * scale))

    image_resized = image.resize(new_size, Image.BILINEAR)

    # Calculate padding needed to reach target size.
    pad_w = tw - new_size[0]
    pad_h = th - new_size[1]
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    image_padded = ImageOps.expand(image_resized, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),  # (C, H, W), values [0,1]
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    tensor = preprocess(image_padded).unsqueeze(0)  # (1, C, H, W)
    tensor = tensor.permute(0, 2, 3, 1)  # Convert to NHWC: (1, H, W, C)
    return tensor.numpy()


def load_tag_mapping(csv_path):
    """
    Load tag mapping from CSV by row order (ignoring header).
    Each row should have columns: tag_id, name, category, count.
    Returns a dictionary where key is row index (starting at 0) and value is a dict with:
        {"name": tag_name, "category": category}
    """
    tag_mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Skip header
        for i, row in enumerate(reader):
            if row:
                # Use the 'name' and 'category' fields.
                tag_mapping[i] = {"name": row[1].strip(), "category": row[2].strip()}
    return tag_mapping


def main(args):
    # Verify model and tag mapping files exist
    model_path = args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    csv_path = args.tag_mapping
    if not csv_path.exists():
        raise FileNotFoundError(f"Tag mapping CSV file not found: {csv_path}")

    # Initialize ONNX session
    session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image_path = args.image_input

    if not image_path.exists():
        raise FileNotFoundError(f"Input image file not found: {image_path}")

    # Preprocess image
    input_tensor = load_and_preprocess_image(image_path, target_size=args.target_size, mean=args.mean, std=args.std)

    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0]  # Expected shape: (1, num_tags)

    if args.temperature != 1.0:
        # Apply temperature scaling to logits
        scaled_logits = logits / args.temperature

        # Compute probabilities using softmax
        exp_scores = np.exp(scaled_logits)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    else:
        # Compute probabilities using sigmoid
        probabilities = 1 / (1 + np.exp(-logits))

    # Load tag mapping
    tag_mapping = load_tag_mapping(args.tag_mapping)

    # Filter predictions using category-specific thresholds
    tags = {}
    sorted_indices = np.argsort(probabilities[0])[::-1]  # All indices sorted in descending order

    for idx in sorted_indices[: args.top_k]:  # Only consider top K predictions
        prob = probabilities[0][idx]
        info = tag_mapping.get(idx)
        if info is None:
            continue

        # Apply category-specific thresholds
        if info["category"] == "0":  # Assuming "0" is for characters
            if prob >= args.character_threshold:
                tags[info["name"]] = prob
        else:
            if prob >= args.general_threshold:
                tags[info["name"]] = prob

    # Print the results
    print("Filtered Predictions:")
    for tag, prob in tags.items():
        print(f"Tag: {tag}, Probability: {prob:.4f}")

    # Save caption if the flag is set
    if args.save_caption:
        output_file = image_path.with_name(image_path.stem + ".txt")

        # Write the output lines to the file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(",".join([tag for tag in tags.keys()]))

        print(f"Predictions saved to: {output_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference on an image using an ONNX model")
    parser.add_argument("image-input", type=Path, help="Path to the input image file")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the ONNX model file")
    parser.add_argument(
        "--target-size", type=int, nargs=2, default=[448, 448], help="Target size (width, height) for resizing"
    )
    parser.add_argument("--mean", type=float, nargs=3, default=[0.5, 0.5, 0.5], help="Mean values for normalization")
    parser.add_argument(
        "--std", type=float, nargs=3, default=[0.5, 0.5, 0.5], help="Standard deviation values for normalization"
    )
    parser.add_argument("--general-threshold", type=float, default=0.35, help="Probability threshold for general tags")
    parser.add_argument(
        "--character-threshold", type=float, default=0.70, help="Probability threshold for character tags"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling for logits. Higher values make predictions more uniform.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of top predictions to consider")
    parser.add_argument("--tag-mapping", type=Path, required=True, help="Path to the CSV file containing tag mapping")
    parser.add_argument(
        "--save-caption", action="store_true", help="Save the predictions as a .txt file next to the image-input"
    )

    args = parser.parse_args()
    main(args)
