from typing import Optional

import bitsandbytes as bnb
import torch
from accelerate import Accelerator
from torch import optim

from .util import parse_dict
from .trainer import OptimizerConfig


def get_optimizer(model, learning_rate: float, optimizer_config: OptimizerConfig):
    optimizer_name = optimizer_config.optimizer_name
    optimizer_args = optimizer_config.optimizer_args

    if optimizer_name == "AdamW":
        optimizer_args = {"lr": learning_rate, **optimizer_args}
        optimizer = optim.AdamW(model.parameters(), **optimizer_args)
        print("Use AdamW optimizer: ", optimizer_args)
    elif optimizer_name == "AdamW8bit":
        optimizer_args = {"lr": learning_rate, **optimizer_args}
        optimizer = bnb.optim.AdamW8bit(model.parameters(), **optimizer_args)
        print("Use AdamW8bit optimizer: ", optimizer_args)
    elif optimizer_name == "Flora":
        from flora_opt.optimizers.torch.flora import Flora

        optimizer_args = {"lr": learning_rate, **optimizer_args}
        optimizer = Flora(model.parameters(), **optimizer_args)
        print("Use Flora optimizer: ", optimizer_args)
    elif optimizer_name == "ProdigyPlusScheduleFree":
        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

        optimizer_args = {"lr": learning_rate, **optimizer_args}
        optimizer = ProdigyPlusScheduleFree(model.parameters(), **optimizer_args)
        print("Use ProdigyPlusScheduleFree optimizer: ", optimizer_args)
    else:
        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

        optimizer_args = {"lr": learning_rate or 1.0, **optimizer_args}
        optimizer = ProdigyPlusScheduleFree(model.parameters(), **optimizer_args)
        print("Use ProdigyPlusScheduleFree optimizer: ", optimizer_args)

    return optimizer


def get_accelerator(args):
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


def get_scheduler(optimizer, training_config, args, **kwargs) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
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

    return argparser, arggroup
