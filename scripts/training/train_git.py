#!/usr/bin/env python3
"""
Refactored GIT model training script using the new library utilities.

This script demonstrates how to use the refactored caption_train library
for training GIT models with a clean, maintainable pipeline.
"""

from pathlib import Path

from caption_train.training.pipeline import create_training_pipeline
from caption_train.utils.arguments import parse_training_args, extract_config_objects, validate_training_args


def main():
    """Main training function using refactored utilities."""
    # Parse arguments using the library utilities for GIT models
    args, groups = parse_training_args("git")

    # Validate arguments
    validate_training_args(args)

    # Extract configuration objects
    configs = extract_config_objects(args, groups)

    training_config = configs["training"]
    peft_config = configs["peft"]
    optimizer_config = configs["optimizer"]
    dataset_config = configs["dataset"]

    # Use model_id from args (with backward compatibility for model_name_or_path)
    model_id = args.model_id or "microsoft/git-base"

    # Determine dataset path (with backward compatibility for train_dir)
    dataset_path = args.dataset or args.dataset_dir or args.train_dir
    if not dataset_path:
        raise ValueError("Either --dataset, --dataset_dir, or --train_dir must be provided")

    dataset_path = Path(dataset_path)

    # Handle legacy block_size argument for GIT models
    if hasattr(args, "block_size") and args.block_size:
        training_config.max_length = args.block_size

    # Handle legacy lora_bits for quantization
    if hasattr(args, "lora_bits") and args.lora_bits < 16:
        training_config.quantize = True

    # Create and run the training pipeline
    pipeline = create_training_pipeline(model_type="git", model_id=model_id)

    print("Starting GIT model training with refactored pipeline...")
    print(f"Model ID: {model_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {dataset_config.output_dir}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Epochs: {training_config.epochs}")
    print(f"Max length: {training_config.max_length}")

    if hasattr(args, "augment_images") and args.augment_images:
        print("Image augmentation: Enabled")

    # Run the complete training pipeline
    pipeline.run_full_pipeline(
        training_config=training_config,
        peft_config=peft_config,
        optimizer_config=optimizer_config,
        dataset_config=dataset_config,
        args=args,
        dataset_path=dataset_path,
    )

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
