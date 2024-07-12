import torch
from typing import Optional
from torch import optim
from accelerate import Accelerator
from flora_opt.optimizers.torch.flora import Flora, FloraAccelerator
import bitsandbytes as bnb


def get_optimizer(model, optimizer_name, training_config):
    if optimizer_name == "AdamW":
        optimizer_args = {
            "lr": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
        }
        optimizer = optim.AdamW(model.parameters(), **optimizer_args)
        print("Use AdamW optimizer: ", optimizer_args)
    elif optimizer_name == "AdamW8bit":
        optimizer_args = {
            "lr": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
        }
        optimizer = bnb.optim.AdamW8bit(model.parameters(), **optimizer_args)
        print("Use AdamW8bit optimizer: ", optimizer_args)
    elif optimizer_name == "Flora":
        optimizer_args = {
            "lr": training_config.learning_rate,
            "rank": training_config.optimizer_rank,
            "relative_step": False,
        }
        optimizer = Flora(model.parameters(), **optimizer_args)
        print("Use Flora optimizer: ", optimizer_args)
    else:
        raise RuntimeError("No optimizer")

    return optimizer


def get_accelerator(args):
    if args.accumulation_rank is not None:
        accelerator_args = {
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "accumulation_compression_rank": args.accumulation_rank,
        }
        accelerator = FloraAccelerator(
            log_with=args.log_with, **accelerator_args
        )
        accelerator.init_trackers(args.name or "caption-train", config=args)
        print("Use FloraAccelerator", accelerator_args)
    else:
        accelerator_args = {
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        accelerator = Accelerator(log_with=args.log_with, **accelerator_args)
        accelerator.init_trackers(args.name or "caption-train", config=args)
        print("Use Accelerator", accelerator_args)

    return accelerator


def get_scheduler(
    optimizer, training_config, args, **kwargs
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
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
    argparser.add_argument(
        "--optimizer_name",
        choices=["AdamW", "AdamW8bit", "Flora"],
        help="Optimizer to use",
    )
    argparser.add_argument(
        "--scheduler",
        default=None,
        choices=["OneCycle"],
        help="Scheduler to use",
    )
    argparser.add_argument(
        "--accumulation_rank",
        type=int,
        help="Rank to use with FloraAccelerator for low rank optimizer",
    )
    argparser.add_argument(
        "-ac",
        "--activation_checkpointing",
        action="store_true",
        help="Activation checkpointing using the FloraAccelerator",
    )
    argparser.add_argument(
        "--accumulation-rank",
        type=int,
        default=None,
        help="Accumulation rank for low rank accumulation",
    )
    argparser.add_argument(
        "--optimizer_rank",
        type=int,
        default=None,
        help="Flora optimizer rank for low-rank optimizer",
    )

    return argparser
