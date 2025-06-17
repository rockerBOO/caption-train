#!/usr/bin/env python3
"""
Refactored BLIP model training script using the new library utilities.

This script demonstrates how to use the refactored caption_train library
for training BLIP models with a clean, maintainable pipeline.
"""

from pathlib import Path

from caption_train.training.pipeline import create_training_pipeline
from caption_train.utils.arguments import parse_training_args, extract_config_objects, validate_training_args


def main():
    """Main training function using refactored utilities."""
    # Parse arguments using the library utilities for BLIP models
    args, groups = parse_training_args("blip")

    # Validate arguments
    validate_training_args(args)

    # Extract configuration objects
    configs = extract_config_objects(args, groups)

    training_config = configs["training"]
    peft_config = configs["peft"]
    optimizer_config = configs["optimizer"]
    dataset_config = configs["dataset"]

    # Use model_id from args
    model_id = args.model_id or "Salesforce/blip-image-captioning-base"

    # Determine dataset path
    dataset_path = args.dataset or args.dataset_dir
    if not dataset_path:
        raise ValueError("Either --dataset or --dataset_dir must be provided")

    dataset_path = Path(dataset_path)

    # Handle weight_decay for BLIP models if provided
    if hasattr(args, "weight_decay") and args.weight_decay:
        if "weight_decay" not in optimizer_config.optimizer_args:
            optimizer_config.optimizer_args["weight_decay"] = args.weight_decay

    # Create and run the training pipeline
    pipeline = create_training_pipeline(model_type="blip", model_id=model_id)

    print("Starting BLIP model training with refactored pipeline...")
    print(f"Model ID: {model_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Output directory: {dataset_config.output_dir}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Epochs: {training_config.epochs}")
    print(f"LoRA rank: {peft_config.rank}")
    print(f"LoRA alpha: {peft_config.alpha}")

    if hasattr(args, "weight_decay"):
        print(f"Weight decay: {args.weight_decay}")

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
