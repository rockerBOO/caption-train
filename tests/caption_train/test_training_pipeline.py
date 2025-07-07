import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from argparse import Namespace

from caption_train.training.pipeline import TrainingPipeline, create_training_pipeline
from caption_train.trainer import TrainingConfig, PeftConfig, OptimizerConfig, FileConfig


@pytest.fixture
def mock_training_config():
    return TrainingConfig(
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=2,
        epochs=1,
        max_length=512,
        shuffle_captions=False,
        frozen_parts=0,
        caption_dropout=0.0,
        quantize=False,
        device="cpu",
        model_id="microsoft/Florence-2-base-ft",
        seed=42,
        log_with="wandb",
        name="test",
        prompt="<MORE_DETAILED_CAPTION>",
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        sample_every_n_epochs=None,
        sample_every_n_steps=None,
        save_every_n_epochs=None,
        save_every_n_steps=None,
    )


@pytest.fixture
def mock_peft_config():
    return PeftConfig(rank=4, alpha=8, rslora=False, target_modules=["qkv", "proj"], init_lora_weights="gaussian")


@pytest.fixture
def mock_optimizer_config():
    return OptimizerConfig(
        optimizer_args={},
        optimizer_name="adamw",
        scheduler="linear",
        accumulation_rank=None,
        activation_checkpointing=None,
        optimizer_rank=None,
        lora_plus_ratio=None,
    )


@pytest.fixture
def mock_file_config():
    return FileConfig(
        dataset_dir=Path("/test/dataset"),
        dataset=None,
        combined_suffix=None,
        generated_suffix=None,
        caption_file_suffix=None,
        recursive=False,
        num_workers=0,
        debug_dataset=False,
    )


@pytest.fixture
def mock_args():
    return Namespace(seed=42, scheduler=True, dataset=Path("/test/dataset"), dataset_dir=None)


def test_training_pipeline_init():
    """Test TrainingPipeline initialization."""
    pipeline = TrainingPipeline()

    assert pipeline.model_type == "florence"
    assert pipeline.model_id == "microsoft/Florence-2-base-ft"
    assert pipeline.target_modules is not None
    assert pipeline.model is None
    assert pipeline.processor is None


def test_training_pipeline_init_with_params():
    """Test TrainingPipeline initialization with parameters."""
    custom_modules = ["custom1", "custom2"]
    pipeline = TrainingPipeline(model_type="git", model_id="microsoft/git-base", target_modules=custom_modules)

    assert pipeline.model_type == "git"
    assert pipeline.model_id == "microsoft/git-base"
    assert pipeline.target_modules == custom_modules


def test_get_default_target_modules_florence():
    """Test getting default target modules for Florence."""
    pipeline = TrainingPipeline(model_type="florence")
    modules = pipeline._get_default_target_modules()

    assert isinstance(modules, list)
    assert len(modules) > 0
    assert "qkv" in modules


def test_get_default_target_modules_git():
    """Test getting default target modules for GIT."""
    pipeline = TrainingPipeline(model_type="git")
    modules = pipeline._get_default_target_modules()

    assert isinstance(modules, list)
    assert "k_proj" in modules
    assert "v_proj" in modules


def test_get_default_target_modules_unknown():
    """Test getting default target modules for unknown model type."""
    with pytest.raises(ValueError, match="Unknown model type"):
        _ = TrainingPipeline(model_type="unknown")


@patch("caption_train.training.pipeline.setup_florence_model")
def test_setup_model_florence(mock_setup_florence, mock_training_config, mock_peft_config):
    """Test setting up Florence model."""
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_setup_florence.return_value = (mock_model, mock_processor)

    pipeline = TrainingPipeline(model_type="florence")
    model, processor = pipeline.setup_model(mock_training_config, mock_peft_config)

    mock_setup_florence.assert_called_once_with(pipeline.model_id, mock_training_config, mock_peft_config)
    assert pipeline.model == mock_model
    assert pipeline.processor == mock_processor
    assert model == mock_model
    assert processor == mock_processor


def test_setup_model_unsupported():
    """Test setting up unsupported model type."""
    with pytest.raises(ValueError, match="Unknown model type"):
        _ = TrainingPipeline(model_type="unsupported")


@patch("caption_train.training.pipeline.get_accelerator")
def test_setup_accelerator(mock_get_accelerator, mock_args):
    """Test setting up accelerator."""
    mock_accelerator = MagicMock()
    mock_accelerator.prepare.return_value = [MagicMock(), MagicMock()]  # Return two objects for unpacking
    mock_get_accelerator.return_value = mock_accelerator

    mock_model = MagicMock()
    mock_processor = MagicMock()

    pipeline = TrainingPipeline()
    pipeline.model = mock_model
    pipeline.processor = mock_processor

    accelerator = pipeline.setup_accelerator(mock_args)

    mock_get_accelerator.assert_called_once_with(mock_args)
    mock_accelerator.prepare.assert_called_once_with(mock_model, mock_processor)
    assert pipeline.accelerator == mock_accelerator
    assert accelerator == mock_accelerator


@patch("caption_train.training.pipeline.get_optimizer")
def test_setup_optimizer(mock_get_optimizer, mock_training_config, mock_optimizer_config):
    """Test setting up optimizer."""
    mock_optimizer = MagicMock()
    mock_get_optimizer.return_value = mock_optimizer

    mock_model = MagicMock()
    mock_accelerator = MagicMock()
    mock_accelerator.prepare.return_value = mock_optimizer

    pipeline = TrainingPipeline()
    pipeline.model = mock_model
    pipeline.accelerator = mock_accelerator

    optimizer = pipeline.setup_optimizer(mock_training_config, mock_optimizer_config)
    assert optimizer is not None

    mock_get_optimizer.assert_called_once_with(mock_model, mock_training_config.learning_rate, mock_optimizer_config)
    mock_accelerator.prepare.assert_called_once_with(mock_optimizer)
    assert pipeline.optimizer == mock_optimizer


def test_setup_optimizer_no_model():
    """Test setting up optimizer without model."""
    pipeline = TrainingPipeline()

    with pytest.raises(ValueError, match="Model must be set up before optimizer"):
        pipeline.setup_optimizer(MagicMock(), MagicMock())


@patch("caption_train.training.pipeline.get_scheduler")
def test_setup_scheduler(mock_get_scheduler, mock_training_config, mock_args):
    """Test setting up scheduler."""
    mock_scheduler = MagicMock()
    mock_get_scheduler.return_value = mock_scheduler

    mock_optimizer = MagicMock()
    mock_accelerator = MagicMock()
    mock_accelerator.prepare.return_value = mock_scheduler

    pipeline = TrainingPipeline()
    pipeline.optimizer = mock_optimizer
    pipeline.accelerator = mock_accelerator

    scheduler = pipeline.setup_scheduler(mock_training_config, mock_args, steps_per_epoch=100)
    assert scheduler is not None

    mock_get_scheduler.assert_called_once_with(mock_optimizer, mock_training_config, mock_args, steps_per_epoch=100)
    assert pipeline.scheduler == mock_scheduler


def test_setup_scheduler_no_scheduler_arg():
    """Test setting up scheduler when scheduler arg is False."""
    args = Namespace(scheduler=False)
    pipeline = TrainingPipeline()
    pipeline.optimizer = MagicMock()

    scheduler = pipeline.setup_scheduler(MagicMock(), args, 100)

    assert scheduler is None
    assert pipeline.scheduler is None


@patch("caption_train.training.pipeline.set_up_image_text_pair")
def test_setup_datasets_directory(mock_set_up_image_text_pair, mock_training_config, mock_file_config):
    """Test setting up datasets with directory path."""
    mock_datasets = MagicMock()
    mock_set_up_image_text_pair.return_value = mock_datasets

    mock_processor = MagicMock()
    mock_accelerator = MagicMock()

    pipeline = TrainingPipeline()
    pipeline.processor = mock_processor
    pipeline.accelerator = mock_accelerator
    pipeline.model = MagicMock()

    dataset_path = Path("/test/dataset")

    with patch("pathlib.Path.is_dir", return_value=True):
        datasets = pipeline.setup_datasets(mock_training_config, mock_file_config, dataset_path)
        assert datasets is not None
        assert datasets is not None, "Dataset setup failed"

    mock_set_up_image_text_pair.assert_called_once()
    mock_datasets.accelerate.assert_called_once_with(mock_accelerator)
    assert pipeline.datasets == mock_datasets


@patch("caption_train.training.pipeline.set_up_datasets")
def test_setup_datasets_file(mock_set_up_datasets, mock_training_config, mock_file_config):
    """Test setting up datasets with file path."""
    mock_datasets = MagicMock()
    mock_set_up_datasets.return_value = mock_datasets

    pipeline = TrainingPipeline()
    pipeline.processor = MagicMock()
    pipeline.accelerator = MagicMock()

    dataset_path = Path("/test/dataset.jsonl")

    with patch("pathlib.Path.is_dir", return_value=False):
        datasets = pipeline.setup_datasets(mock_training_config, mock_file_config, dataset_path)
        assert datasets is not None
        assert datasets is not None, "Dataset setup failed"

    mock_set_up_datasets.assert_called_once()
    assert pipeline.datasets == mock_datasets


@patch("caption_train.training.pipeline.Trainer")
def test_setup_trainer(mock_trainer_class, mock_training_config, mock_file_config):
    """Test setting up trainer."""
    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer

    pipeline = TrainingPipeline()
    pipeline.model = MagicMock()
    pipeline.processor = MagicMock()
    pipeline.optimizer = MagicMock()
    pipeline.scheduler = MagicMock()
    pipeline.accelerator = MagicMock()
    pipeline.datasets = MagicMock()

    trainer = pipeline.setup_trainer(mock_training_config, mock_file_config)

    mock_trainer_class.assert_called_once()
    assert pipeline.trainer == mock_trainer
    assert trainer == mock_trainer


def test_setup_trainer_missing_components():
    """Test setting up trainer with missing components."""
    pipeline = TrainingPipeline()

    with pytest.raises(ValueError, match="All components must be set up before trainer"):
        pipeline.setup_trainer(MagicMock(), MagicMock())


def test_run_training():
    """Test running training."""
    mock_trainer = MagicMock()

    pipeline = TrainingPipeline()
    pipeline.trainer = mock_trainer

    pipeline.run_training()

    mock_trainer.train.assert_called_once()


def test_run_training_no_trainer():
    """Test running training without trainer."""
    pipeline = TrainingPipeline()

    with pytest.raises(ValueError, match="Trainer must be set up before running training"):
        pipeline.run_training()


def test_create_training_pipeline():
    """Test factory function for creating training pipeline."""
    pipeline = create_training_pipeline()

    assert isinstance(pipeline, TrainingPipeline)
    assert pipeline.model_type == "florence"
    assert pipeline.model_id == "microsoft/Florence-2-base-ft"


def test_create_training_pipeline_with_params():
    """Test factory function with custom parameters."""
    pipeline = create_training_pipeline(model_type="git", model_id="microsoft/git-base")

    assert pipeline.model_type == "git"
    assert pipeline.model_id == "microsoft/git-base"
