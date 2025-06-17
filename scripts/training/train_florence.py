#!/usr/bin/env python3
"""
Refactored Florence-2 training script using the new library utilities.

This script demonstrates how to use the refactored caption_train library
for a clean, maintainable training pipeline.
"""

from pathlib import Path

from caption_train.training.pipeline import create_training_pipeline
from caption_train.utils.arguments import parse_training_args, extract_config_objects, validate_training_args


def main():
    """Main training function using refactored utilities."""
    # Parse arguments using the library utilities
    args, groups = parse_training_args("florence")

    # Validate arguments
    validate_training_args(args)

    # Extract configuration objects
    configs = extract_config_objects(args, groups)

    training_config = configs["training"]
    peft_config = configs["peft"]
    optimizer_config = configs["optimizer"]
    dataset_config = configs["dataset"]

    # Determine dataset path
    dataset_path = args.dataset or args.dataset_dir
    if not dataset_path:
        raise ValueError("Either --dataset or --dataset_dir must be provided")

    dataset_path = Path(dataset_path)

    # Create and run the training pipeline
    pipeline = create_training_pipeline(model_type="florence", model_id=args.model_id)

    print("Starting Florence-2 training with refactored pipeline...")
    print(f"Model ID: {args.model_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {dataset_config.output_dir}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Epochs: {training_config.epochs}")

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
