import pytest
import torch
import argparse
from unittest.mock import MagicMock
from caption_train.opt import get_optimizer, lora_plus, get_accelerator, get_scheduler, opt_config_args
from caption_train.trainer import OptimizerConfig


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 5)
        self.layer2 = torch.nn.Linear(5, 2)

        # Setup some parameters with requires_grad
        for name, param in self.named_parameters():
            param.requires_grad = True


def test_get_optimizer_adamw():
    model = MockModel()
    config = OptimizerConfig(
        optimizer_name="AdamW",
        optimizer_args={},
        scheduler=None,
        accumulation_rank=None,
        activation_checkpointing=False,
        optimizer_rank=None,
        lora_plus_ratio=None,
    )

    optimizer = get_optimizer(model, 0.001, config)

    assert isinstance(optimizer, torch.optim.AdamW)


@pytest.mark.parametrize("optimizer_name", ["ProdigyPlusScheduleFree"])
def test_get_optimizer_complex(optimizer_name, monkeypatch):
    if optimizer_name == "ProdigyPlusScheduleFree":
        monkeypatch.setattr(
            "prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree", lambda *args, **kwargs: MagicMock()
        )

    model = MockModel()
    config = OptimizerConfig(
        optimizer_name=optimizer_name,
        optimizer_args={},
        scheduler=None,
        accumulation_rank=None,
        activation_checkpointing=False,
        optimizer_rank=None,
        lora_plus_ratio=None,
    )

    optimizer = get_optimizer(model, 0.001, config)

    assert optimizer is not None


def test_lora_plus():
    model = MockModel()
    parameters = list(model.named_parameters())

    grouped_parameters = lora_plus(parameters, learning_rate=0.001, ratio=16)

    assert len(grouped_parameters) == 2
    assert grouped_parameters[0]["lr"] == 0.001
    assert grouped_parameters[1]["lr"] == 0.016


def test_get_accelerator():
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    # Add some mock attributes to simulate different cases
    args.log_with = None
    args.name = "test-run"
    args.gradient_accumulation_steps = 1

    accelerator = get_accelerator(args)

    assert accelerator is not None


def test_get_scheduler_with_real_optimizer():
    # Create a real optimizer
    model = MockModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Create a mock Namespace with OneCycle scheduler
    class MockArgs:
        scheduler = "OneCycle"

    class MockTrainingConfig:
        learning_rate = 0.001
        epochs = 10

    scheduler = get_scheduler(optimizer, MockTrainingConfig(), MockArgs(), steps_per_epoch=10)

    assert scheduler is not None
    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)


def test_opt_config_args():
    parser = argparse.ArgumentParser()
    updated_parser, group = opt_config_args(parser)

    # Check that arguments have been added
    assert updated_parser is not None

    # Use parse_known_args to check argument parsing works
    parsed_args, _ = updated_parser.parse_known_args(
        ["--optimizer_name", "AdamW", "--scheduler", "OneCycle", "--optimizer_args", "{'momentum': 0.9}"]
    )

    assert parsed_args.optimizer_name == "AdamW"
    assert parsed_args.scheduler == "OneCycle"
