"""Enhanced argument parsing utilities for training scripts."""

import argparse
from pathlib import Path
from typing import Any

from caption_train.trainer import (
    peft_config_args,
    training_config_args,
)
from caption_train.datasets import datasets_config_args
from caption_train.opt import opt_config_args
from caption_train.utils.config import merge_args_with_config


def create_base_parser(
    description: str = "Model training script", formatter_class: type = argparse.RawTextHelpFormatter
) -> argparse.ArgumentParser:
    """Create base argument parser with common arguments.

    This function creates a foundational ArgumentParser that contains arguments
    common to all training scripts. It serves as the base for model-specific
    parsers created by create_florence_parser(), create_git_parser(), etc.

    Common Arguments Added:
        --config: Path to TOML configuration file for loading training parameters
        --model_id: HuggingFace model identifier (e.g., "microsoft/Florence-2-base-ft")
        --dataset: Path to dataset directory or file
        --output_dir: Output directory for saving trained models (required)

    Usage Pattern:
        This function is typically called by model-specific parser creators:
        ```python
        parser = create_base_parser("Florence-2 training script")
        # Add model-specific arguments...
        parser, peft_group = peft_config_args(parser, target_modules)
        parser, training_group = training_config_args(parser)
        ```

    Configuration File Support:
        The --config argument allows users to specify TOML files containing
        training parameters. Config files are merged with command-line arguments,
        with CLI args taking precedence over config file values.

    Args:
        description: Description text shown in help output
                    Defaults to "Model training script"
        formatter_class: argparse formatter class for help text formatting
                        Defaults to RawTextHelpFormatter for better multi-line help

    Returns:
        ArgumentParser: Configured parser with base arguments

    Example:
        ```python
        # Create parser for custom training script
        parser = create_base_parser("My custom training script")

        # Parse arguments
        args = parser.parse_args([
            "--model_id", "microsoft/Florence-2-base-ft",
            "--dataset", "/path/to/images",
            "--output_dir", "./checkpoints",
            "--config", "config.toml"
        ])

        # Access parsed values
        print(f"Model: {args.model_id}")
        print(f"Dataset: {args.dataset}")
        print(f"Config: {args.config}")
        ```
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)

    # Add common arguments
    parser.add_argument("--config", type=Path, help="Path to TOML configuration file")

    parser.add_argument(
        "--model_id", type=str, default="microsoft/Florence-2-base-ft", help="HuggingFace model identifier"
    )

    parser.add_argument("--dataset", type=Path, help="Path to dataset directory or file")

    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for saving models")

    return parser


def create_florence_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse._ArgumentGroup]]:
    """Create argument parser for Florence-2 training.

    This function creates a complete ArgumentParser configured specifically for
    Florence-2 model training. It combines the base parser with model-specific
    argument groups for PEFT, training, optimization, and dataset configuration.

    Florence-2 Specific Features:
        - Uses FLORENCE_TARGET_MODULES as default LoRA target modules
        - Optimized for vision-language model fine-tuning
        - Supports task-specific prompts like "<MORE_DETAILED_CAPTION>"
        - Configured for typical Florence-2 training patterns

    Argument Groups Created:
        "peft": PEFT/LoRA configuration (rank, alpha, target_modules, etc.)
        "training": Core training parameters (learning rate, batch size, epochs)
        "opt": Optimizer and scheduler configuration
        "dataset": Dataset loading and processing options

    Target Modules:
        Florence-2 uses specific target modules optimized for vision-language tasks.
        The default target modules are imported from caption_train.models.florence
        and include key attention and projection layers.

    Common Usage:
        ```python
        parser, groups = create_florence_parser()
        args = parser.parse_args()

        # Extract configuration objects
        configs = extract_config_objects(args, groups)
        training_config = configs["training"]
        peft_config = configs["peft"]
        ```

    Integration with Training Pipeline:
        The parser is designed to work seamlessly with TrainingPipeline:
        ```python
        args, groups = parse_training_args("florence")
        configs = extract_config_objects(args, groups)

        pipeline = create_training_pipeline("florence", args.model_id)
        pipeline.run_full_pipeline(
            training_config=configs["training"],
            peft_config=configs["peft"],
            optimizer_config=configs["optimizer"],
            dataset_config=configs["dataset"],
            args=args,
            dataset_path=args.dataset
        )
        ```

    Returns:
        tuple[ArgumentParser, dict[str, ArgumentGroup]]:
            - parser: Complete ArgumentParser ready for parsing Florence-2 training args
            - groups_dict: Dictionary mapping group names to ArgumentGroup objects
                          Keys: "peft", "training", "opt", "dataset"

    Example:
        ```python
        # Create Florence-2 parser
        parser, groups = create_florence_parser()

        # Parse typical Florence-2 training arguments
        args = parser.parse_args([
            "--model_id", "microsoft/Florence-2-base-ft",
            "--dataset", "/path/to/images",
            "--output_dir", "./checkpoints",
            "--rank", "8",
            "--alpha", "16",
            "--learning_rate", "1e-4",
            "--batch_size", "4",
            "--epochs", "5",
            "--prompt", "<MORE_DETAILED_CAPTION>"
        ])

        # Access Florence-2 specific arguments
        print(f"LoRA rank: {args.rank}")
        print(f"Task prompt: {args.prompt}")
        print(f"Target modules: {args.target_modules}")
        ```
    """
    from caption_train.models.florence import FLORENCE_TARGET_MODULES

    parser = create_base_parser("Florence-2 model training")

    # Add argument groups
    parser, peft_group = peft_config_args(parser, FLORENCE_TARGET_MODULES)
    parser, training_group = training_config_args(parser)
    parser, opt_group = opt_config_args(parser)
    parser, dataset_group = datasets_config_args(parser)

    groups = {
        "peft": peft_group,
        "training": training_group,
        "opt": opt_group,
        "dataset": dataset_group,
    }

    return parser, groups


def create_git_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse._ArgumentGroup]]:
    """Create argument parser for GIT model training.

    This function creates a complete ArgumentParser configured specifically for
    GIT (GenerativeImage2Text) model training. It includes GIT-specific arguments
    and legacy argument aliases for backward compatibility.

    GIT Model Specific Features:
        - Custom target modules for GIT's transformer architecture
        - Backward compatibility aliases (model_name_or_path, train_dir)
        - Image augmentation support for data preprocessing
        - Configurable sequence length (block_size)
        - LoRA quantization options (4, 8, 16 bit)

    Target Modules:
        GIT uses attention mechanism target modules:
        - k_proj, v_proj, q_proj: Key, value, query projections
        - out_proj: Output projection
        - query, key, value: Additional attention components

    Legacy Compatibility:
        Supports aliases for backward compatibility with existing scripts:
        - --model_name_or_path: Alias for --model_id
        - --train_dir: Alias for --dataset_dir
        These are handled by parse_training_args() automatically.

    GIT-Specific Arguments:
        --augment_images: Enable image augmentations during training
        --block_size: Maximum sequence length (default: 2048)
        --lora_bits: Quantization bits for LoRA (4, 8, or 16)
        --model_name_or_path: Legacy alias for model_id
        --train_dir: Legacy alias for dataset directory

    Usage Pattern:
        ```python
        parser, groups = create_git_parser()
        args = parser.parse_args()

        # Legacy arguments are automatically mapped
        model_id = args.model_id or args.model_name_or_path
        dataset_path = args.dataset or args.train_dir
        ```

    Returns:
        tuple[ArgumentParser, dict[str, ArgumentGroup]]:
            - parser: Complete ArgumentParser for GIT model training
            - groups_dict: Dictionary with argument groups
                          Keys: "peft", "training", "opt", "dataset"

    Example:
        ```python
        # Create GIT parser
        parser, groups = create_git_parser()

        # Parse GIT training arguments (using legacy names)
        args = parser.parse_args([
            "--model_name_or_path", "microsoft/git-base",
            "--train_dir", "/path/to/images",
            "--output_dir", "./checkpoints",
            "--augment_images",
            "--block_size", "1024",
            "--lora_bits", "4",
            "--rank", "8",
            "--learning_rate", "2e-4"
        ])

        # Access GIT-specific arguments
        print(f"Augmentation: {args.augment_images}")
        print(f"Block size: {args.block_size}")
        print(f"LoRA bits: {args.lora_bits}")

        # Legacy arguments are available
        print(f"Model (legacy): {args.model_name_or_path}")
        print(f"Dataset (legacy): {args.train_dir}")
        ```
    """
    git_target_modules = [
        "k_proj",
        "v_proj",
        "q_proj",
        "out_proj",
        "query",
        "key",
        "value",
    ]

    parser = create_base_parser("GIT model training")

    # Add GIT-specific arguments
    parser.add_argument(
        "--model_name_or_path", type=str, help="Model name from hugging face or path to model (alias for model_id)"
    )

    parser.add_argument("--train_dir", type=Path, help="Directory with training data (alias for dataset_dir)")

    parser.add_argument("--augment_images", action="store_true", help="Apply image augmentations during training")

    parser.add_argument("--block_size", type=int, default=2048, help="Maximum sequence length")

    parser.add_argument(
        "--lora_bits", type=int, choices=[4, 8, 16], default=16, help="Quantization bits for LoRA (4, 8, or 16)"
    )

    # Add argument groups
    parser, peft_group = peft_config_args(parser, git_target_modules)
    parser, training_group = training_config_args(parser)
    parser, opt_group = opt_config_args(parser)
    parser, dataset_group = datasets_config_args(parser)

    groups = {
        "peft": peft_group,
        "training": training_group,
        "opt": opt_group,
        "dataset": dataset_group,
    }

    return parser, groups


def create_blip_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse._ArgumentGroup]]:
    """Create argument parser for BLIP model training.

    This function creates a complete ArgumentParser configured specifically for
    BLIP (Bootstrapping Language-Image Pre-training) model training. It includes
    BLIP-specific optimization parameters and target modules.

    BLIP Model Specific Features:
        - Custom target modules optimized for BLIP's architecture
        - Weight decay parameter for regularization
        - Support for both BlipForConditionalGeneration and AutoModelForVision2Seq
        - Optimized for image captioning and visual question answering tasks

    Target Modules:
        BLIP uses specific target modules imported from caption_train.models.blip:
        - q_proj, v_proj, k_proj: Query, value, key projections
        - out_proj: Output projection layer
        - fc1, fc2: Feed-forward network layers
        These modules are carefully selected for BLIP's transformer architecture.

    BLIP-Specific Arguments:
        --weight_decay: Weight decay for optimizer regularization (default: 1e-4)
                       BLIP models often benefit from moderate weight decay

    Optimization Considerations:
        BLIP models typically require:
        - Moderate weight decay (1e-4) for stable training
        - Careful learning rate selection (often 1e-5 to 5e-5)
        - Appropriate batch size based on GPU memory (2-8 typical)

    Usage with Training Pipeline:
        ```python
        args, groups = parse_training_args("blip")
        configs = extract_config_objects(args, groups)

        pipeline = create_training_pipeline("blip", args.model_id)
        # Weight decay is automatically handled by optimizer config
        ```

    Returns:
        tuple[ArgumentParser, dict[str, ArgumentGroup]]:
            - parser: Complete ArgumentParser for BLIP model training
            - groups_dict: Dictionary with argument groups
                          Keys: "peft", "training", "opt", "dataset"

    Example:
        ```python
        # Create BLIP parser
        parser, groups = create_blip_parser()

        # Parse BLIP training arguments
        args = parser.parse_args([
            "--model_id", "Salesforce/blip-image-captioning-base",
            "--dataset", "/path/to/images",
            "--output_dir", "./checkpoints",
            "--weight_decay", "1e-5",  # BLIP-specific
            "--rank", "8",
            "--alpha", "16",
            "--learning_rate", "2e-5",  # Conservative for BLIP
            "--batch_size", "4",
            "--epochs", "3"
        ])

        # Access BLIP-specific arguments
        print(f"Weight decay: {args.weight_decay}")
        print(f"Target modules: {args.target_modules}")

        # Extract configs for training
        configs = extract_config_objects(args, groups)
        optimizer_config = configs["optimizer"]
        # Weight decay is included in optimizer_config.optimizer_args
        ```
    """
    from caption_train.models.blip import BLIP_TARGET_MODULES

    parser = create_base_parser("BLIP model training")

    # Add BLIP-specific arguments
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")

    # Add argument groups
    parser, peft_group = peft_config_args(parser, BLIP_TARGET_MODULES)
    parser, training_group = training_config_args(parser)
    parser, opt_group = opt_config_args(parser)
    parser, dataset_group = datasets_config_args(parser)

    groups = {
        "peft": peft_group,
        "training": training_group,
        "opt": opt_group,
        "dataset": dataset_group,
    }

    return parser, groups


def parse_training_args(
    model_type: str = "florence", args: list[str] | None = None
) -> tuple[argparse.Namespace, dict[str, argparse._ArgumentGroup]]:
    """Parse training arguments for a specific model type.

    This is the main entry point for parsing command-line arguments in training
    scripts. It automatically selects the appropriate parser based on model type,
    handles legacy argument aliases, merges configuration files, and returns
    both parsed arguments and argument groups for further processing.

    Supported Model Types:
        "florence": Florence-2 vision-language models
        "git": GIT (GenerativeImage2Text) models
        "blip": BLIP (Bootstrapping Language-Image Pre-training) models

    Automatic Processing:
        1. Creates model-specific parser using create_*_parser()
        2. Parses command-line arguments
        3. Handles legacy argument aliases for backward compatibility
        4. Merges TOML configuration file if --config is provided
        5. Returns parsed arguments and groups for config object extraction

    Legacy Argument Handling:
        Automatically maps legacy argument names to current names:
        - model_name_or_path → model_id
        - train_dir → dataset_dir
        This ensures backward compatibility with existing training scripts.

    Configuration File Integration:
        If --config is provided, the TOML configuration file is automatically
        loaded and merged with command-line arguments. CLI arguments take
        precedence over config file values.

    Integration with extract_config_objects():
        The returned argument groups are designed to work with extract_config_objects()
        to create strongly-typed configuration objects:
        ```python
        args, groups = parse_training_args("florence")
        configs = extract_config_objects(args, groups)
        ```

    Error Handling:
        - Raises ValueError for unsupported model types
        - Argument parsing errors are handled by argparse (exits with error message)
        - Configuration file errors are handled by merge_args_with_config()

    Args:
        model_type: Type of model to train
                   Supported: "florence", "git", "blip"
                   Default: "florence"
        args: List of command-line arguments to parse
             If None, uses sys.argv (normal command-line parsing)
             If provided, parses the given list (useful for testing)

    Returns:
        tuple[Namespace, dict[str, ArgumentGroup]]:
            - parsed_args: argparse.Namespace with all parsed arguments
                          Includes legacy argument mapping and config file merging
            - argument_groups: Dictionary mapping group names to ArgumentGroup objects
                              Used by extract_config_objects() to create config objects

    Raises:
        ValueError: If model_type is not supported

    Example:
        ```python
        # Parse Florence-2 arguments from command line
        args, groups = parse_training_args("florence")

        # Parse specific arguments (useful for testing)
        test_args = ["--model_id", "microsoft/Florence-2-base-ft", "--epochs", "5"]
        args, groups = parse_training_args("florence", test_args)

        # Access parsed arguments
        print(f"Model ID: {args.model_id}")
        print(f"Epochs: {args.epochs}")

        # Create configuration objects
        configs = extract_config_objects(args, groups)
        training_config = configs["training"]
        peft_config = configs["peft"]

        # Use with training pipeline
        pipeline = create_training_pipeline("florence", args.model_id)
        pipeline.run_full_pipeline(
            training_config=training_config,
            peft_config=peft_config,
            optimizer_config=configs["optimizer"],
            dataset_config=configs["dataset"],
            args=args,
            dataset_path=args.dataset
        )
        ```

    Legacy Compatibility Example:
        ```python
        # These legacy arguments are automatically mapped
        legacy_args = [
            "--model_name_or_path", "microsoft/git-base",  # → model_id
            "--train_dir", "/path/to/data",                 # → dataset_dir
            "--output_dir", "./checkpoints"
        ]

        args, groups = parse_training_args("git", legacy_args)

        # Both legacy and new names are available
        assert args.model_id == "microsoft/git-base"
        assert args.model_name_or_path == "microsoft/git-base"
        assert args.dataset_dir == "/path/to/data"
        assert args.train_dir == "/path/to/data"
        ```
    """
    if model_type == "florence":
        parser, groups = create_florence_parser()
    elif model_type == "git":
        parser, groups = create_git_parser()
    elif model_type == "blip":
        parser, groups = create_blip_parser()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    parsed_args = parser.parse_args(args)

    # Handle legacy argument aliases for backward compatibility
    if hasattr(parsed_args, "model_name_or_path") and parsed_args.model_name_or_path:
        if not hasattr(parsed_args, "model_id") or not parsed_args.model_id:
            parsed_args.model_id = parsed_args.model_name_or_path

    if hasattr(parsed_args, "train_dir") and parsed_args.train_dir:
        if not hasattr(parsed_args, "dataset_dir") or not parsed_args.dataset_dir:
            parsed_args.dataset_dir = parsed_args.train_dir

    # Merge with config file if provided
    if hasattr(parsed_args, "config") and parsed_args.config:
        parsed_args = merge_args_with_config(parsed_args, parsed_args.config)

    return parsed_args, groups


def extract_config_objects(args: argparse.Namespace, groups: dict[str, argparse._ArgumentGroup]) -> dict[str, Any]:
    """Extract configuration objects from parsed arguments and argument groups.

    This function takes parsed command-line arguments and their corresponding
    argument groups, then creates strongly-typed configuration objects for
    different aspects of training (training params, PEFT, optimizer, dataset).

    The function uses the `get_group_args` utility from `caption_train.util` to
    extract only the arguments that belong to each specific group, then creates
    the appropriate configuration dataclass instance.

    Argument Group Mapping:
        "training" -> TrainingConfig: Core training parameters (LR, batch size, etc.)
        "peft" -> PeftConfig: LoRA/PEFT parameters (rank, alpha, target modules)
        "opt" -> OptimizerConfig: Optimizer and scheduler configuration
        "dataset" -> FileConfig: Dataset loading and file handling configuration

    Dependencies:
        - Imports TrainingConfig, PeftConfig, OptimizerConfig, FileConfig from caption_train.trainer
        - Uses get_group_args from caption_train.util to extract group-specific arguments
        - Requires that all configuration classes have matching field names to argument names

    Error Handling:
        - Missing required arguments will cause TypeError when creating config objects
        - Invalid argument types will cause TypeError during dataclass instantiation
        - Missing groups in the groups dict will be silently skipped (optional groups)

    Args:
        args: Parsed arguments namespace from argparse.ArgumentParser.parse_args()
        groups: Dictionary mapping group names to argparse._ArgumentGroup objects
                Typically created by parse_training_args() or similar functions

    Returns:
        Dictionary mapping configuration type names to config object instances:
        {
            "training": TrainingConfig(...),     # if "training" group exists
            "peft": PeftConfig(...),             # if "peft" group exists
            "optimizer": OptimizerConfig(...),   # if "opt" group exists
            "dataset": FileConfig(...),          # if "dataset" group exists
        }

    Example:
        ```python
        # Typical usage via parse_training_args
        args, groups = parse_training_args("florence", ["--epochs", "5", "--batch_size", "4"])
        configs = extract_config_objects(args, groups)

        # Access specific configurations
        training_config = configs["training"]  # TrainingConfig instance
        peft_config = configs["peft"]          # PeftConfig instance

        # Use with training pipeline
        pipeline = create_training_pipeline("florence", "microsoft/Florence-2-base-ft")
        pipeline.run_full_pipeline(
            training_config=configs["training"],
            peft_config=configs["peft"],
            optimizer_config=configs["optimizer"],
            dataset_config=configs["dataset"],
            args=args,
            dataset_path=Path("/path/to/dataset")
        )
        ```
    """
    from caption_train.util import get_group_args
    from caption_train.trainer import TrainingConfig, PeftConfig, OptimizerConfig, FileConfig

    configs = {}

    if "training" in groups:
        configs["training"] = TrainingConfig(**get_group_args(args, groups["training"]))

    if "peft" in groups:
        configs["peft"] = PeftConfig(**get_group_args(args, groups["peft"]))

    if "opt" in groups:
        configs["optimizer"] = OptimizerConfig(**get_group_args(args, groups["opt"]))

    if "dataset" in groups:
        configs["dataset"] = FileConfig(**get_group_args(args, groups["dataset"]))

    return configs


def create_inference_parser() -> argparse.ArgumentParser:
    """Create argument parser for inference scripts.
    
    This function creates an ArgumentParser specifically designed for model
    inference scripts. It includes arguments for model loading, input/output
    paths, and generation parameters for caption generation or other inference tasks.
    
    Inference Arguments:
        --model_path: Path to trained model (required)
        --input_path: Path to input image or directory (required)
        --output_path: Path to save generated outputs (optional)
        --batch_size: Batch size for inference (default: 1)
    
    Generation Parameters:
        --max_length: Maximum caption/output length (default: 256)
        --num_beams: Number of beams for beam search (default: 1)
        --do_sample: Use sampling instead of beam search (flag)
        --temperature: Sampling temperature for randomness (default: 1.0)
        --top_p: Top-p sampling parameter for nucleus sampling (default: 1.0)
    
    Usage Patterns:
        Single image inference:
        ```bash
        python inference.py --model_path ./checkpoint --input_path image.jpg
        ```
        
        Batch inference on directory:
        ```bash
        python inference.py --model_path ./checkpoint --input_path ./images/ --batch_size 4
        ```
        
        Custom generation parameters:
        ```bash
        python inference.py --model_path ./checkpoint --input_path image.jpg \
            --do_sample --temperature 0.8 --top_p 0.9 --max_length 128
        ```
    
    Generation Strategy:
        - Default: Greedy decoding (num_beams=1, do_sample=False)
        - Beam search: Set num_beams > 1
        - Sampling: Set do_sample=True, adjust temperature and top_p

    Returns:
        ArgumentParser: Configured parser for inference scripts
        
    Example:
        ```python
        # Create inference parser
        parser = create_inference_parser()
        
        # Parse typical inference arguments
        args = parser.parse_args([
            "--model_path", "./trained_model",
            "--input_path", "./test_images",
            "--output_path", "./captions.txt",
            "--batch_size", "8",
            "--max_length", "256",
            "--num_beams", "3",
            "--do_sample",
            "--temperature", "0.7",
            "--top_p", "0.9"
        ])
        
        # Access inference parameters
        print(f"Model: {args.model_path}")
        print(f"Input: {args.input_path}")
        print(f"Batch size: {args.batch_size}")
        print(f"Generation: beams={args.num_beams}, sample={args.do_sample}")
        print(f"Sampling: temp={args.temperature}, top_p={args.top_p}")
        ```
    """
    parser = argparse.ArgumentParser(
        description="Model inference script", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--model_path", type=Path, required=True, help="Path to trained model")

    parser.add_argument("--input_path", type=Path, required=True, help="Path to input image or directory")

    parser.add_argument("--output_path", type=Path, help="Path to save generated captions")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")

    parser.add_argument("--max_length", type=int, default=256, help="Maximum caption length")

    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")

    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of beam search")

    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")

    return parser


def validate_training_args(args: argparse.Namespace) -> None:
    """Validate training arguments and raise errors for invalid combinations.

    This function performs comprehensive validation of parsed arguments to catch
    common errors early and provide helpful error messages. It checks for:
    - Required argument presence
    - File/directory existence
    - Valid parameter ranges
    - Logical argument combinations

    Validation Rules:
        1. Dataset Requirements: Either `dataset` OR `dataset_dir` must be provided
           - Only checked if these attributes exist on the args object
           - Supports both None values and missing attributes gracefully

        2. Path Existence: If provided, dataset paths must exist
           - `dataset`: Must be a valid file path
           - `dataset_dir`: Must be a valid directory path

        3. Parameter Ranges:
           - `learning_rate`: Must be positive (> 0)
           - `batch_size`: Must be positive (> 0)
           - `epochs`: Must be positive (> 0)

    Usage Patterns:
        # Called automatically by parse_training_args()
        args, groups = parse_training_args("florence")
        # validation happens here automatically

        # Manual validation for custom argument parsing
        parser = argparse.ArgumentParser()
        # ... add arguments ...
        args = parser.parse_args()
        validate_training_args(args)  # Explicit validation

    Common Validation Errors:
        - "Either --dataset or --dataset_dir must be provided"
        - "Dataset path does not exist: /path/to/file"
        - "Dataset directory does not exist: /path/to/dir"
        - "Learning rate must be positive"
        - "Batch size must be positive"
        - "Number of epochs must be positive"

    Args:
        args: Parsed arguments namespace from argparse
              Can contain any subset of validation-relevant attributes

    Raises:
        ValueError: If validation fails with descriptive error message

    Example:
        ```python
        # This will pass validation
        args = Namespace(
            dataset="/path/to/existing/file.jsonl",
            dataset_dir=None,
            learning_rate=1e-4,
            batch_size=4,
            epochs=5
        )
        validate_training_args(args)  # No exception

        # This will raise ValueError
        args = Namespace(dataset=None, dataset_dir=None)
        validate_training_args(args)  # ValueError: Either --dataset or --dataset_dir must be provided
        ```
    """
    # Check that required paths exist
    if hasattr(args, "dataset") and args.dataset and not Path(args.dataset).exists():
        raise ValueError(f"Dataset path does not exist: {args.dataset}")

    if hasattr(args, "dataset_dir") and args.dataset_dir and not Path(args.dataset_dir).exists():
        raise ValueError(f"Dataset directory does not exist: {args.dataset_dir}")

    # Check that either dataset or dataset_dir is provided (only if they are attributes)
    if hasattr(args, "dataset") or hasattr(args, "dataset_dir"):
        dataset = getattr(args, "dataset", None)
        dataset_dir = getattr(args, "dataset_dir", None)
        if not (dataset or dataset_dir):
            raise ValueError("Either --dataset or --dataset_dir must be provided")

    # Validate learning rate
    if hasattr(args, "learning_rate") and args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    # Validate batch size
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    # Validate epochs
    if hasattr(args, "epochs") and args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
