#!/usr/bin/env python3
"""
Refactored inference script using the new library utilities.

This script demonstrates how to use the refactored caption_train library
for clean, maintainable inference.
"""

import torch
from pathlib import Path
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor
from peft import PeftModel

from caption_train.inference.sampling import sample_with_prompts
from caption_train.utils.arguments import create_inference_parser


def load_model_and_processor(model_path: Path, base_model_id: str = "microsoft/Florence-2-base-ft"):
    """Load trained model and processor."""
    print(f"Loading model from {model_path}")

    # Load base model and processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Load PEFT model
    model = PeftModel.from_pretrained(base_model_id, model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    return model, processor


def load_images(input_path: Path) -> list[Image.Image]:
    """Load images from path (file or directory)."""
    if input_path.is_file():
        # Single image
        return [Image.open(input_path)]
    elif input_path.is_dir():
        # Directory of images
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            images.extend(input_path.glob(ext))
        return [Image.open(img_path) for img_path in sorted(images)]
    else:
        raise ValueError(f"Input path not found: {input_path}")


def batch_images(images: list[Image.Image], batch_size: int) -> list[list[Image.Image]]:
    """Split images into batches."""
    return [images[i : i + batch_size] for i in range(0, len(images), batch_size)]


def save_captions(captions: list[str], output_path: Path) -> None:
    """Save generated captions to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, caption in enumerate(captions):
            f.write(f"Image {i + 1}: {caption}\n")

    print(f"Captions saved to {output_path}")


def main():
    """Main inference function."""
    parser = create_inference_parser()
    args = parser.parse_args()

    # Validate arguments
    if not args.model_path.exists():
        raise ValueError(f"Model path not found: {args.model_path}")
    if not args.input_path.exists():
        raise ValueError(f"Input path not found: {args.input_path}")

    # Set up accelerator
    accelerator = Accelerator()

    # Load model and processor
    model, processor = load_model_and_processor(args.model_path)
    model = accelerator.prepare(model)
    model.eval()

    print(f"Model loaded successfully on device: {accelerator.device}")

    # Load images
    images = load_images(args.input_path)
    print(f"Loaded {len(images)} images")

    # Process images in batches
    all_captions = []
    image_batches = batch_images(images, args.batch_size)

    print("Generating captions...")
    for batch_idx, image_batch in enumerate(image_batches):
        print(f"Processing batch {batch_idx + 1}/{len(image_batches)}")

        # Convert images to tensors (this would normally be done by processor)
        # For simplicity, we'll use the sampling utility directly
        captions = sample_with_prompts(
            model=model,
            processor=processor,
            accelerator=accelerator,
            images=image_batch,
            prompts=None,  # No specific prompts
            max_length=args.max_length,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        all_captions.extend(captions)

        # Print some examples
        for i, caption in enumerate(captions):
            print(f"  Image {batch_idx * args.batch_size + i + 1}: {caption}")

    print(f"\nGenerated {len(all_captions)} captions")

    # Save captions if output path specified
    if args.output_path:
        save_captions(all_captions, args.output_path)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
