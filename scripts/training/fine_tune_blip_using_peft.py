#!/usr/bin/env python3
"""
Refactored fine-tune-blip-using-peft.py script using the new library utilities.

This script demonstrates how to use the refactored caption_train library
for fine-tuning BLIP models with PEFT on external datasets like the football dataset.
"""

import argparse
from pathlib import Path

from caption_train.training.pipeline import create_training_pipeline
from caption_train.trainer import TrainingConfig, PeftConfig, OptimizerConfig, FileConfig
from caption_train.models.blip import BLIP_TARGET_MODULES


def create_finetune_parser() -> argparse.ArgumentParser:
    """Create argument parser compatible with fine-tune-blip-using-peft.py script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune BLIP model using PEFT on external datasets"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_id",
        default="Salesforce/blip-image-captioning-base",
        help="HuggingFace model identifier"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="./training/caption",
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--dataset_name",
        default="ybelkada/football-dataset",
        help="HuggingFace dataset name"
    )
    
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train"
    )
    
    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=10,
        help="Sample model output every n steps"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=32,
        help="LoRA alpha"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=[
            "self.query",
            "self.key", 
            "self.value",
            "output.dense",
            "self_attn.qkv",
            "self_attn.projection",
            "mlp.fc1",
            "mlp.fc2",
        ],
        help="Target modules for LoRA"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to train on (cuda/cpu)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens for generation during sampling"
    )
    
    return parser


def main():
    """Main training function using refactored utilities."""
    # Parse arguments
    parser = create_finetune_parser()
    args = parser.parse_args()
    
    # Create a mock args object that matches what our refactored utilities expect
    refactored_args = type('Args', (), {})()
    refactored_args.model_id = args.model_id
    refactored_args.dataset = args.dataset_name
    refactored_args.output_dir = args.output_dir
    refactored_args.learning_rate = args.learning_rate
    refactored_args.batch_size = args.batch_size
    refactored_args.epochs = args.epochs
    refactored_args.gradient_accumulation_steps = 1
    refactored_args.gradient_checkpointing = False
    refactored_args.shuffle_captions = False
    refactored_args.frozen_parts = 0
    refactored_args.caption_dropout = 0.0
    refactored_args.seed = args.seed
    refactored_args.rank = args.rank
    refactored_args.alpha = args.alpha
    refactored_args.dropout = args.dropout
    refactored_args.weight_decay = args.weight_decay
    refactored_args.save_every_n_epochs = None
    refactored_args.save_every_n_steps = None
    refactored_args.sample_every_n_epochs = None
    refactored_args.sample_every_n_steps = args.sample_every_n_steps
    refactored_args.quantize = False
    refactored_args.device = args.device
    refactored_args.log_with = None
    refactored_args.name = None
    refactored_args.prompt = None
    refactored_args.max_length = None
    refactored_args.scheduler = None
    refactored_args.recursive = True
    refactored_args.num_workers = 0
    refactored_args.debug_dataset = False
    refactored_args.target_modules = args.target_modules
    refactored_args.rslora = False
    refactored_args.init_lora_weights = "gaussian"
    refactored_args.optimizer_name = "AdamW"
    refactored_args.optimizer_args = {"weight_decay": args.weight_decay}
    refactored_args.max_new_tokens = args.max_new_tokens
    
    # Create configurations
    training_config = TrainingConfig(
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=None,
        shuffle_captions=False,
        frozen_parts=0,
        caption_dropout=0.0,
        quantize=False,
        device=args.device,
        model_id=args.model_id,
        seed=args.seed,
        log_with=None,
        name=None,
        prompt=None,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        sample_every_n_epochs=None,
        sample_every_n_steps=args.sample_every_n_steps,
        save_every_n_epochs=None,
        save_every_n_steps=None
    )
    
    peft_config = PeftConfig(
        rank=args.rank,
        alpha=args.alpha,
        rslora=False,
        target_modules=args.target_modules,
        init_lora_weights="gaussian"
    )
    
    optimizer_config = OptimizerConfig(
        optimizer_args={"weight_decay": args.weight_decay},
        optimizer_name="AdamW",
        scheduler=None,
        accumulation_rank=None,
        activation_checkpointing=None,
        optimizer_rank=None,
        lora_plus_ratio=None
    )
    
    dataset_config = FileConfig(
        dataset_dir=None,
        dataset=args.dataset_name,
        combined_suffix=None,
        generated_suffix=None,
        caption_file_suffix=None,
        recursive=True,
        num_workers=0,
        debug_dataset=False
    )
    dataset_config.output_dir = args.output_dir
    
    # Create and run the training pipeline
    pipeline = create_training_pipeline(
        model_type="blip",
        model_id=args.model_id
    )
    
    print("Starting BLIP fine-tuning with PEFT using refactored pipeline...")
    print(f"Model ID: {args.model_id}")
    print(f"Dataset: {args.dataset_name} (split: {args.split})")
    print(f"Output directory: {args.output_dir}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.rank}")
    print(f"LoRA alpha: {args.alpha}")
    print(f"LoRA dropout: {args.dropout}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Target modules: {args.target_modules}")
    print(f"Sample every {args.sample_every_n_steps} steps")
    print(f"Max new tokens for generation: {args.max_new_tokens}")
    
    # Run the complete training pipeline
    pipeline.run_full_pipeline(
        training_config=training_config,
        peft_config=peft_config,
        optimizer_config=optimizer_config,
        dataset_config=dataset_config,
        args=refactored_args,
        dataset_path=Path(args.dataset_name)  # For external datasets, use the dataset name
    )
    
    print(f"Fine-tuning completed successfully! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()