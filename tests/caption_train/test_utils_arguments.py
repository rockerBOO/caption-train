import pytest
import tempfile
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch, MagicMock

from caption_train.utils.arguments import (
    create_base_parser,
    create_florence_parser,
    create_git_parser,
    parse_training_args,
    extract_config_objects,
    create_inference_parser,
    validate_training_args,
)


def test_create_base_parser():
    """Test creating base argument parser."""
    parser = create_base_parser()

    # Test parsing some basic arguments
    args = parser.parse_args(["--model_id", "test-model", "--output_dir", "/test/output"])

    assert args.model_id == "test-model"
    assert args.output_dir == Path("/test/output")
    assert args.config is None
    assert args.dataset is None


def test_create_base_parser_with_config():
    """Test base parser with config file."""
    parser = create_base_parser("Test description")

    args = parser.parse_args(
        [
            "--config",
            "/path/to/config.toml",
            "--model_id",
            "test-model",
            "--dataset",
            "/path/to/dataset",
            "--output_dir",
            "/test/output",
        ]
    )

    assert args.config == Path("/path/to/config.toml")
    assert args.dataset == Path("/path/to/dataset")


def test_create_florence_parser():
    """Test creating Florence-2 parser."""
    parser, groups = create_florence_parser()

    # Test that all expected groups are present
    assert "peft" in groups
    assert "training" in groups
    assert "opt" in groups
    assert "dataset" in groups

    # Test parsing some Florence-specific arguments
    args = parser.parse_args(
        [
            "--model_id",
            "microsoft/Florence-2-base-ft",
            "--output_dir",
            "/test/output",
            "--rank",
            "8",
            "--alpha",
            "16",
            "--learning_rate",
            "1e-4",
        ]
    )

    assert args.model_id == "microsoft/Florence-2-base-ft"
    assert args.rank == 8
    assert args.alpha == 16
    assert args.learning_rate == 1e-4


def test_create_git_parser():
    """Test creating GIT parser."""
    parser, groups = create_git_parser()

    # Test that all expected groups are present
    assert "peft" in groups
    assert "training" in groups
    assert "opt" in groups
    assert "dataset" in groups

    # Test parsing GIT-specific arguments
    args = parser.parse_args(
        [
            "--model_id",
            "microsoft/git-base",
            "--output_dir",
            "/test/output",
            "--augment_images",
            "--block_size",
            "1024",
            "--lora_bits",
            "4",
        ]
    )

    assert args.model_id == "microsoft/git-base"
    assert args.augment_images is True
    assert args.block_size == 1024
    assert args.lora_bits == 4


def test_parse_training_args_florence():
    """Test parsing training arguments for Florence model."""
    test_args = ["--model_id", "microsoft/Florence-2-base-ft", "--output_dir", "/test/output", "--rank", "4"]

    args, groups = parse_training_args("florence", test_args)

    assert args.model_id == "microsoft/Florence-2-base-ft"
    assert args.rank == 4
    assert "peft" in groups


def test_parse_training_args_git():
    """Test parsing training arguments for GIT model."""
    test_args = ["--model_id", "microsoft/git-base", "--output_dir", "/test/output", "--lora_bits", "8"]

    args, groups = parse_training_args("git", test_args)

    assert args.model_id == "microsoft/git-base"
    assert args.lora_bits == 8


def test_parse_training_args_unsupported():
    """Test parsing training arguments for unsupported model type."""
    with pytest.raises(ValueError, match="Unsupported model type"):
        parse_training_args("unsupported", [])


@patch("caption_train.utils.arguments.merge_args_with_config")
def test_parse_training_args_with_config(mock_merge_args):
    """Test parsing training arguments with config file."""
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write('[model]\nname = "test"')
        config_path = Path(f.name)

    try:
        mock_merge_args.return_value = Namespace(model_id="test", output_dir=Path("/test"))

        test_args = [
            "--config",
            str(config_path),
            "--model_id",
            "microsoft/Florence-2-base-ft",
            "--output_dir",
            "/test/output",
        ]

        args, groups = parse_training_args("florence", test_args)

        # Verify merge_args_with_config was called
        mock_merge_args.assert_called_once()

    finally:
        config_path.unlink()


@patch("caption_train.util.get_group_args")
def test_extract_config_objects(mock_get_group_args):
    """Test extracting configuration objects from parsed arguments."""

    # Create a dictionary mapping group objects to their data
    training_data = {
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "batch_size": 4,
        "epochs": 1,
        "max_length": 512,
        "shuffle_captions": False,
        "frozen_parts": 0,
        "caption_dropout": 0.0,
        "quantize": False,
        "device": "cpu",
        "model_id": "test-model",
        "seed": 42,
        "log_with": "wandb",
        "name": "test",
        "prompt": None,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 1,
        "sample_every_n_epochs": None,
        "sample_every_n_steps": None,
        "save_every_n_epochs": None,
        "save_every_n_steps": None,
    }

    peft_data = {
        "rank": 8,
        "alpha": 16,
        "rslora": False,
        "target_modules": ["q_proj", "v_proj"],
        "init_lora_weights": "gaussian",
    }

    opt_data = {
        "optimizer_args": {},
        "optimizer_name": "adamw",
        "scheduler": None,
        "accumulation_rank": None,
        "activation_checkpointing": None,
        "optimizer_rank": None,
        "lora_plus_ratio": None,
    }

    dataset_data = {
        "dataset_dir": Path("/test"),
        "dataset": None,
        "combined_suffix": None,
        "generated_suffix": None,
        "caption_file_suffix": None,
        "recursive": False,
        "num_workers": 0,
        "debug_dataset": False,
    }

    args = Namespace()

    # Create mock groups
    training_group = MagicMock()
    peft_group = MagicMock()
    opt_group = MagicMock()
    dataset_group = MagicMock()

    groups = {"training": training_group, "peft": peft_group, "opt": opt_group, "dataset": dataset_group}

    # Mock get_group_args to return specific data for specific group objects
    def mock_get_group_args_side_effect(args, group):
        if group is training_group:
            return training_data
        elif group is peft_group:
            return peft_data
        elif group is opt_group:
            return opt_data
        elif group is dataset_group:
            return dataset_data
        return {}

    mock_get_group_args.side_effect = mock_get_group_args_side_effect

    configs = extract_config_objects(args, groups)

    # Verify all config objects were created
    assert "training" in configs
    assert "peft" in configs
    assert "optimizer" in configs
    assert "dataset" in configs

    # Verify get_group_args was called for each group
    assert mock_get_group_args.call_count == 4


def test_create_inference_parser():
    """Test creating inference argument parser."""
    parser = create_inference_parser()

    args = parser.parse_args(
        [
            "--model_path",
            "/path/to/model",
            "--input_path",
            "/path/to/input",
            "--output_path",
            "/path/to/output",
            "--batch_size",
            "4",
            "--max_length",
            "128",
            "--num_beams",
            "3",
            "--do_sample",
            "--temperature",
            "0.8",
            "--top_p",
            "0.9",
        ]
    )

    assert args.model_path == Path("/path/to/model")
    assert args.input_path == Path("/path/to/input")
    assert args.output_path == Path("/path/to/output")
    assert args.batch_size == 4
    assert args.max_length == 128
    assert args.num_beams == 3
    assert args.do_sample is True
    assert args.temperature == 0.8
    assert args.top_p == 0.9


def test_validate_training_args_valid():
    """Test validating valid training arguments."""
    # Create temporary dataset directory
    with tempfile.TemporaryDirectory() as temp_dir:
        args = Namespace(dataset=Path(temp_dir), dataset_dir=None, learning_rate=1e-4, batch_size=4, epochs=10)

        # Should not raise any exception
        validate_training_args(args)


def test_validate_training_args_missing_dataset():
    """Test validating arguments with missing dataset path."""
    args = Namespace(dataset=Path("/nonexistent/path"), dataset_dir=None)

    with pytest.raises(ValueError, match="Dataset path does not exist"):
        validate_training_args(args)


def test_validate_training_args_missing_dataset_dir():
    """Test validating arguments with missing dataset directory."""
    args = Namespace(dataset=None, dataset_dir=Path("/nonexistent/path"))

    with pytest.raises(ValueError, match="Dataset directory does not exist"):
        validate_training_args(args)


def test_validate_training_args_no_dataset_specified():
    """Test validating arguments with no dataset specified."""
    args = Namespace(dataset=None, dataset_dir=None)

    with pytest.raises(ValueError, match="Either --dataset or --dataset_dir must be provided"):
        validate_training_args(args)


def test_validate_training_args_invalid_learning_rate():
    """Test validating arguments with invalid learning rate."""
    with tempfile.TemporaryDirectory() as temp_dir:
        args = Namespace(dataset=Path(temp_dir), dataset_dir=None, learning_rate=-1e-4)

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            validate_training_args(args)


def test_validate_training_args_invalid_batch_size():
    """Test validating arguments with invalid batch size."""
    with tempfile.TemporaryDirectory() as temp_dir:
        args = Namespace(dataset=Path(temp_dir), dataset_dir=None, batch_size=0)

        with pytest.raises(ValueError, match="Batch size must be positive"):
            validate_training_args(args)


def test_validate_training_args_invalid_epochs():
    """Test validating arguments with invalid epochs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        args = Namespace(dataset=Path(temp_dir), dataset_dir=None, epochs=-5)

        with pytest.raises(ValueError, match="Number of epochs must be positive"):
            validate_training_args(args)


def test_validate_training_args_missing_attributes():
    """Test validating arguments with missing attributes."""
    # This should not raise errors for missing optional attributes
    args = Namespace()

    # Should not raise any exception when attributes are missing
    validate_training_args(args)
