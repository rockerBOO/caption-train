import pytest
import torch
import argparse
from unittest.mock import MagicMock
from pathlib import Path
from accelerate.optimizer import AcceleratedOptimizer
from accelerate import Accelerator
from transformers import AutoProcessor
from contextlib import contextmanager

from caption_train.trainer import (
    TrainingConfig,
    FileConfig,
    Trainer,
    peft_config_args,
    training_config_args,
    flora_config_args,
)


@pytest.fixture
def mock_training_config():
    return TrainingConfig(
        dropout=0.25,
        learning_rate=1e-4,
        batch_size=1,
        epochs=5,
        max_length=None,
        shuffle_captions=False,
        frozen_parts=0,
        caption_dropout=0.0,
        quantize=False,
        device="cpu",
        model_id="microsoft/Florence-2-base-ft",
        seed=42,
        log_with="wandb",
        name="test_training",
        prompt=None,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        sample_every_n_epochs=None,
        sample_every_n_steps=None,
        save_every_n_epochs=None,
        save_every_n_steps=None,
    )


@pytest.fixture
def mock_trainer(mock_training_config):
    # Create mock objects
    model = MagicMock()
    processor = MagicMock(spec=AutoProcessor)
    optimizer = MagicMock(spec=AcceleratedOptimizer)
    scheduler = MagicMock()
    accelerator = MagicMock(spec=Accelerator)
    datasets = MagicMock()
    file_config = FileConfig(
        dataset_dir=Path("/tmp/dataset"),
        dataset=None,
        combined_suffix=None,
        generated_suffix=None,
        caption_file_suffix=None,
        recursive=False,
        num_workers=0,
        debug_dataset=False,
    )

    # Add output_dir attribute for save_model test
    file_config.output_dir = Path("/tmp/output")

    # Additional setup for mocks
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.train = MagicMock()
    model.generate = MagicMock()
    model.save_pretrained = MagicMock()

    # Create a mock output object with a loss attribute
    class MockOutput:
        def __init__(self, loss):
            self.loss = loss

    # Simulate the model's __call__ method
    def mock_call(*args, **kwargs):
        return MockOutput(torch.tensor(0.5))

    model.side_effect = mock_call
    model.__call__ = MagicMock(side_effect=mock_call)

    optimizer.train = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    optimizer.optimizer = MagicMock()

    # Simulate the autocast context manager
    @contextmanager
    def mock_autocast():
        yield

    accelerator.autocast = mock_autocast
    accelerator.backward = MagicMock()
    accelerator.log = MagicMock()
    accelerator.wait_for_everyone = MagicMock()
    accelerator.is_main_process = True
    accelerator.save = MagicMock()
    accelerator.trackers = ["wandb"]
    accelerator.get_tracker = MagicMock(return_value=MagicMock())
    accelerator.accumulate = MagicMock()
    accelerator.unwrap_model = MagicMock(return_value=model)
    accelerator.sync_gradients = True

    datasets.train_dataloader = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.rand(1, 3, 224, 224),
            "labels": torch.tensor([[4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "text": ["Test caption"],
        }
    ]

    trainer = Trainer(
        model=model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        datasets=datasets,
        config=mock_training_config,
        file_config=file_config,
    )

    return trainer


def test_process_batch(mock_trainer):
    # Prepare a mock batch
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.rand(1, 3, 224, 224),
        "labels": torch.tensor([[4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    # Call process_batch
    outputs, loss = mock_trainer.process_batch(batch)

    # Verify calls and returns
    mock_trainer.model.assert_called_once_with(
        input_ids=batch["input_ids"].to(mock_trainer.model.device),
        pixel_values=batch["pixel_values"].to(mock_trainer.model.device, dtype=mock_trainer.model.dtype),
        labels=batch["labels"].to(mock_trainer.model.device),
        attention_mask=batch["attention_mask"].to(mock_trainer.model.device),
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.item() == 0.5


def test_optimizer_logs(mock_trainer):
    # Setup the optimizer with a mock get_dlr method
    mock_group = {"params": [torch.tensor([1.0])]}
    mock_trainer.optimizer.optimizer.param_groups = [mock_group]
    mock_trainer.optimizer.optimizer.get_dlr = MagicMock(return_value=0.01)

    # Call optimizer_logs
    logs = mock_trainer.optimizer_logs()

    # Verify the returned logs
    assert "lr/d*lr-group0" in logs
    assert logs["lr/d*lr-group0"] == 0.01


def test_start_training(mock_trainer):
    # Call start_training
    mock_trainer.start_training()

    # Verify model is set to train mode
    mock_trainer.model.train.assert_called_once()

    # Verify wandb metrics are defined
    wandb_tracker = mock_trainer.accelerator.get_tracker("wandb")
    wandb_tracker.define_metric.assert_called()


def test_save_model(mock_trainer):
    # Call save_model
    mock_trainer.save_model()

    # Verify calls
    mock_trainer.accelerator.wait_for_everyone.assert_called_once()
    mock_trainer.accelerator.unwrap_model.assert_called_once()
    mock_trainer.model.save_pretrained.assert_called_once()


def test_peft_config_args():
    parser = argparse.ArgumentParser()
    target_modules = ["q_proj", "v_proj"]

    updated_parser, group = peft_config_args(parser, target_modules)

    # Verify parser is updated
    assert updated_parser is parser

    # Parse known arguments
    parsed_args, _ = updated_parser.parse_known_args(
        [
            "--target_modules",
            "linear1 linear2",
            "--rslora",
            "--rank",
            "8",
            "--alpha",
            "16",
            "--init_lora_weights",
            "eva",
        ]
    )

    # Verify argument parsing
    assert parsed_args.target_modules == "linear1 linear2"
    assert parsed_args.rslora is True
    assert parsed_args.rank == 8
    assert parsed_args.alpha == 16
    assert parsed_args.init_lora_weights == "eva"


def test_training_config_args():
    parser = argparse.ArgumentParser()

    updated_parser, group = training_config_args(parser)

    # Verify parser is updated
    assert updated_parser is parser

    # Parse known arguments
    parsed_args, _ = updated_parser.parse_known_args(
        [
            "--learning_rate",
            "2e-4",
            "--batch_size",
            "4",
            "--epochs",
            "10",
            "--gradient_checkpointing",
            "--quantize",
            "--device",
            "cuda",
            "--seed",
            "123",
            "--log_with",
            "wandb",
            "--name",
            "test_run",
            "--shuffle_captions",
            "--caption_dropout",
            "0.1",
        ]
    )

    # Verify argument parsing
    assert parsed_args.learning_rate == 2e-4
    assert parsed_args.batch_size == 4
    assert parsed_args.epochs == 10
    assert parsed_args.gradient_checkpointing is True
    assert parsed_args.quantize is True
    assert parsed_args.device == "cuda"
    assert parsed_args.seed == 123
    assert parsed_args.log_with == "wandb"
    assert parsed_args.name == "test_run"
    assert parsed_args.shuffle_captions is True
    assert float(parsed_args.caption_dropout) == 0.1


def test_flora_config_args():
    parser = argparse.ArgumentParser()

    updated_parser, group = flora_config_args(parser)

    # Verify parser is updated
    assert updated_parser is parser

    # Parse known arguments
    parsed_args, _ = updated_parser.parse_known_args(["--accumulation_compression"])

    # Verify argument parsing
    assert parsed_args.accumulation_compression is True
