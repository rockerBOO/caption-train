import argparse

import torch
from accelerate import Accelerator
from torch import optim

from .util import parse_dict
from .trainer import OptimizerConfig


def get_optimizer(model: torch.nn.Module, learning_rate: float, optimizer_config: OptimizerConfig):
    optimizer_name = optimizer_config.optimizer_name
    optimizer_args = optimizer_config.optimizer_args

    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]

    if optimizer_name == "AdamW":
        optimizer_args = {"lr": learning_rate, **optimizer_args}
        optimizer = optim.AdamW(parameters, **optimizer_args)
        print("Use AdamW optimizer: ", optimizer_args)
    elif optimizer_name == "AdamW8bit":
        import bitsandbytes as bnb

        optimizer_args = {"lr": learning_rate, **optimizer_args}
        optimizer = bnb.optim.AdamW8bit(parameters, **optimizer_args)
        print("Use AdamW8bit optimizer: ", optimizer_args)
    elif optimizer_name == "Flora":
        from flora_opt.optimizers.torch.flora import Flora

        optimizer_args = {"lr": learning_rate, **optimizer_args}
        optimizer = Flora(parameters, **optimizer_args)
        print("Use Flora optimizer: ", optimizer_args)
    elif optimizer_name == "ProdigyPlusScheduleFree":
        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

        if optimizer_config.lora_plus_ratio is not None:
            parameters = lora_plus(parameters, learning_rate, optimizer_config.lora_plus_ratio)

        optimizer_args = {"lr": learning_rate, **optimizer_args}

        optimizer = ProdigyPlusScheduleFree(parameters, **optimizer_args)
        print("Use ProdigyPlusScheduleFree optimizer: ", optimizer_args)
    else:
        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

        optimizer_args = {"lr": learning_rate or 1.0, **optimizer_args}
        optimizer = ProdigyPlusScheduleFree(parameters, **optimizer_args)

        print("Use ProdigyPlusScheduleFree optimizer: ", optimizer_args)

    return optimizer


def lora_plus(parameters: list[tuple[str, torch.Tensor]], learning_rate: float, ratio: int = 16):
    lora_parameters: list[torch.Tensor] = []
    lora_plus_parameters: list[torch.Tensor] = []
    for name, param in parameters:
        if "lora_B" in name:
            lora_plus_parameters.append(param)
        else:
            lora_parameters.append(param)

    grouped_parameters = [
        {"params": lora_parameters, "lr": learning_rate},
        {"params": lora_plus_parameters, "lr": learning_rate * ratio},
    ]
    return grouped_parameters


def get_accelerator(args: argparse.Namespace):
    if hasattr(args, "accumulation_rank") and args.accumulation_rank is not None:
        accelerator_args = {
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "accumulation_compression_rank": args.accumulation_rank,
        }
        from flora_opt.optimizers.torch.flora import FloraAccelerator

        accelerator = FloraAccelerator(log_with=args.log_with, **accelerator_args)
        accelerator.init_trackers(args.name or "caption-train", config=args)
        print("Use FloraAccelerator", accelerator_args)
    else:
        accelerator_args = {}

        if hasattr(args, "gradient_accumulation_steps"):
            accelerator_args["gradient_accumulation_steps"] = args.gradient_accumulation_steps

        if hasattr(args, "log_with"):
            accelerator_args["log_with"] = args.log_with

        name = "caption-train"

        if hasattr(args, "name"):
            name = args.name

        accelerator = Accelerator(**accelerator_args)
        accelerator.init_trackers(name, config=args)
        print("Use Accelerator", accelerator_args)

    return accelerator


def get_scheduler(
    optimizer: torch.optim.Optimizer, training_config, args, **kwargs
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if args.scheduler == "OneCycle":
        scheduler_args = {
            "max_lr": training_config.learning_rate,
            "epochs": training_config.epochs,
            "cycle_momentum": False,
            "base_momentum": 0.85,
            "max_momentum": 0.95,
        }
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            **scheduler_args,
            **kwargs,
        )
        print("Use OneCycle scheduler: ", {**scheduler_args, **kwargs})
        return scheduler
    else:
        return None


def opt_config_args(argparser):
    arggroup = argparser.add_argument_group("Optimizer")
    arggroup.add_argument(
        "--optimizer_name",
        help="Optimizer to use",
    )
    arggroup.add_argument(
        "--scheduler",
        default=None,
        choices=["OneCycle"],
        help="Scheduler to use",
    )
    arggroup.add_argument(
        "--accumulation_rank",
        type=int,
        help="Rank to use with FloraAccelerator for low rank optimizer",
    )
    arggroup.add_argument(
        "-ac",
        "--activation_checkpointing",
        action="store_true",
        help="Activation checkpointing using the FloraAccelerator",
    )
    arggroup.add_argument(
        "--accumulation-rank",
        type=int,
        default=None,
        help="Accumulation rank for low rank accumulation",
    )
    arggroup.add_argument(
        "--optimizer_rank",
        type=int,
        default=None,
        help="Flora optimizer rank for low-rank optimizer",
    )
    arggroup.add_argument("--optimizer_args", type=parse_dict, default={}, help="Optimizer args")
    arggroup.add_argument("--lora_plus_ratio", type=int, default=None, help="Ratio to use with LoRA+")

    return argparser, arggroup
