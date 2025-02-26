import argparse
from pathlib import Path
from accelerate.optimizer import AcceleratedOptimizer
from peft.peft_model import PeftModelForCausalLM
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List
import torch
from accelerate import Accelerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from caption_train.util import LossRecorder
from caption_train.datasets import Datasets


@dataclass
class TrainingConfig:
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    max_length: Optional[int]
    shuffle_captions: bool
    frozen_parts: int
    caption_dropout: float
    quantize: bool
    device: str
    model_id: str
    seed: int
    log_with: str
    name: str
    prompt: str
    gradient_checkpointing: bool
    gradient_accumulation_steps: int
    sample_every_n_epochs: Optional[int]
    sample_every_n_steps: Optional[int]
    save_every_n_epochs: Optional[int]
    save_every_n_steps: Optional[int]


@dataclass
class PeftConfig:
    rank: int
    alpha: int
    rslora: bool
    target_modules: List[str]


@dataclass
class OptimizerConfig:
    optimizer_args: dict
    optimizer_name: str
    scheduler: str
    accumulation_rank: Optional[int]
    activation_checkpointing: Optional[bool]
    optimizer_rank: Optional[int]


@dataclass
class FileConfig:
    output_dir: Path
    dataset_dir: Optional[Path]
    dataset: Optional[Path]
    combined_suffix: Optional[str]
    generated_suffix: Optional[str]
    caption_file_suffix: Optional[str]
    recursive: bool
    num_workers: int
    debug_dataset: bool


@dataclass
class Trainer:
    model: torch.nn.Module
    processor: AutoProcessor
    optimizer: AcceleratedOptimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    accelerator: Accelerator
    datasets: Datasets
    config: TrainingConfig
    file_config: FileConfig

    def process_batch(self, batch):
        with self.accelerator.autocast():
            device = self.model.device
            dtype = self.model.dtype
            outputs = self.model(
                input_ids=batch["input_ids"].to(device),
                pixel_values=batch["pixel_values"].to(device, dtype=dtype),
                labels=batch["labels"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = outputs.loss

        return outputs, loss

    def optimizer_logs(self) -> dict[str, float]:
        logs = {}
        optimizer = self.optimizer.optimizer
        if hasattr(optimizer, "get_dlr"):
            for i, group in enumerate(optimizer.param_groups):
                logs[f"lr/d*lr-group{i}"] = optimizer.get_dlr(group)

        return logs

    def start_training(self):
        self.model.train()
        if "wandb" in self.accelerator.trackers:
            import wandb

            wandb_tracker = self.accelerator.get_tracker("wandb")

            wandb_tracker.define_metric("*", step_metric="global_step")
            wandb_tracker.define_metric("global_step", hidden=True)

    def train(self):
        self.start_training()
        loss_recorder = LossRecorder()

        step = 0

        progress = tqdm(
            total=(len(self.datasets.train_dataloader) // self.config.gradient_accumulation_steps) * self.config.epochs
        )
        for epoch in range(self.config.epochs):
            epoch = epoch + 1
            self.accelerator.print("Epoch:", epoch)
            for idx, batch in enumerate(self.datasets.train_dataloader):
                self.optimizer.train()
                with self.accelerator.accumulate(self.model):
                    outputs, loss = self.process_batch(batch)

                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler:
                        self.scheduler.step()

                    loss_recorder.add(
                        epoch=epoch - 1,
                        step=idx,
                        loss=loss.detach().item(),
                    )

                if self.accelerator.sync_gradients:
                    step += 1
                    progress.set_postfix({"avg_loss": loss_recorder.moving_average, **self.optimizer_logs()})
                    progress.update()

                    self.accelerator.log(
                        {"avg_loss": loss_recorder.moving_average, **self.optimizer_logs(), "global_step": step}
                    )

                    # Sample
                    if self.config.sample_every_n_steps is not None and step % self.config.sample_every_n_steps == 0:
                        self.optimizer.eval()
                        self.sample(batch, labels)

                    # Save
                    if self.config.save_every_n_steps is not None and step % self.config.save_every_n_steps == 0:
                        self.save_model(f"step-{step}")

            # Sample
            if self.config.sample_every_n_epochs is not None and epoch % self.config.sample_every_n_epochs == 0:
                self.optimizer.eval()
                self.sample(batch, labels)

            # Save
            if self.config.save_every_n_epochs is not None and epoch % self.config.save_every_n_epochs == 0:
                self.save_model(f"epoch-{epoch}")

        self.save_model()

    @torch.no_grad()
    def sample(self, batch, texts):
        with self.accelerator.autocast():
            generated_output = self.model.generate(**batch)

        decoded = self.processor.batch_decode(
            generated_output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for gen, cap in zip(decoded, texts):
            print(f"Gen: {gen}")
            print(f"Cap: {cap}")

    def save_model(self, name=None):
        # hugging face models saved to the directory
        self.accelerator.wait_for_everyone()
        unwrapped_model: PeftModelForCausalLM = self.accelerator.unwrap_model(self.model)
        if name is not None:
            name = "_" + str(name)
        else:
            name = ""
        unwrapped_model.save_pretrained(
            str(Path(self.file_config.output_dir) / name),
            save_function=self.accelerator.save,
        )

        print(f"Saved to {str(Path(self.file_config.output_dir) / name)}")


def peft_config_args(argparser: argparse.ArgumentParser, target_modules):
    arggroup = argparser.add_argument_group("PEFT")

    arggroup.add_argument(
        "--target_modules",
        nargs="+",
        type=list,
        default=target_modules,
        help=f"Target modules to be trained. Consider Linear and Conv2D modules in the base model. Default: {' '.join(target_modules)}.",
    )
    arggroup.add_argument(
        "--rslora",
        action="store_true",
        help="RS LoRA scales alpha to size of rank",
    )
    arggroup.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Rank/dim for the LoRA. Default: 4",
    )
    arggroup.add_argument(
        "--alpha",
        type=float,
        default=4,
        help="Alpha for scaling the LoRA weights. Default: 4",
    )
    return argparser, arggroup


def training_config_args(argparser):
    arggroup = argparser.add_argument_group("Training")
    arggroup.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout for the LoRA network. Default: 0.25",
    )
    arggroup.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the LoRA. Default: 1e-4",
    )
    arggroup.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the image/caption pairs. Default: 1",
    )
    arggroup.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    arggroup.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to run. Default: 5",
    )
    arggroup.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Gradient checkpointing to reduce memory usage in exchange for slower training",
    )
    arggroup.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the training model to 4-bit",
    )
    arggroup.add_argument(
        "--sample_every_n_epochs",
        type=int,
        help="Sample the dataset every n epochs",
    )
    arggroup.add_argument(
        "--sample_every_n_steps",
        type=int,
        help="Sample the dataset every n steps",
    )
    arggroup.add_argument(
        "--save_every_n_epochs",
        type=int,
        help="Save the model every n epochs",
    )
    arggroup.add_argument(
        "--save_every_n_steps",
        type=int,
        help="Save the model every n steps",
    )
    arggroup.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the training on. Default: cuda or cpu",
    )
    arggroup.add_argument(
        "--model_id",
        default="microsoft/Florence-2-base-ft",
        help="Model to train on. microsoft/Florence-2-base-ft or microsoft/Florence-2-large-ft. Default: microsoft/Florence-2-base-ft",
    )
    arggroup.add_argument("--seed", type=int, default=None, help="Seed used for random numbers")
    arggroup.add_argument(
        "--log_with",
        choices=["all", "wandb", "tensorboard"],
        help="Log with. all, wandb, tensorboard",
    )
    arggroup.add_argument("--name", help="Name to be used with saving and logging")
    arggroup.add_argument(
        "--shuffle_captions",
        action="store_true",
        help="Shuffle captions when training",
    )
    arggroup.add_argument(
        "--frozen_parts",
        default=0,
        help="How many parts (parts separated by ',') do we want to keep in place when shuffling",
    )
    arggroup.add_argument(
        "--caption_dropout",
        default=0.0,
        help="Amount of parts we dropout in the caption.",
    )
    arggroup.add_argument(
        "--max_length",
        default=None,
        type=int,
        help="Max length of the input text. Defaults to max length of the Florence 2 processor.",
    )
    arggroup.add_argument("--prompt", type=str, default=None, help="Task for florence")

    return argparser, arggroup


def flora_config_args(argparser):
    arggroup = argparser.add_argument_group("Flora")
    arggroup.add_argument(
        "--accumulation_compression",
        action="store_true",
        help="Accumulation compression for FloraAccelerator",
    )

    return argparser, arggroup
