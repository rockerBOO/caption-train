import argparse
from pathlib import Path
from accelerate.optimizer import AcceleratedOptimizer
from peft.peft_model import PeftModelForCausalLM
from tqdm import tqdm
from dataclasses import dataclass
from typing import Literal
import torch
from accelerate import Accelerator
from transformers import AutoProcessor

from caption_train.util import LossRecorder
from caption_train.datasets import Datasets


@dataclass
class TrainingConfig:
    """Configuration for training parameters.
    
    This class holds all training-specific parameters used during model fine-tuning.
    All fields are required unless explicitly marked as optional (with | None).
    
    Core Training Parameters:
        dropout: LoRA dropout rate (0.0-1.0, typically 0.1-0.25)
        learning_rate: Learning rate for optimizer (typically 1e-4 to 1e-5)
        batch_size: Number of samples per batch (adjust based on GPU memory)
        epochs: Number of training epochs
        gradient_accumulation_steps: Steps to accumulate before optimizer update
        gradient_checkpointing: Whether to use gradient checkpointing (saves memory)
    
    Model Configuration:
        model_id: HuggingFace model identifier (e.g., "microsoft/Florence-2-base-ft")
        device: Device to train on ("cuda", "cpu", or specific device like "cuda:0")
        quantize: Whether to use 4-bit quantization to reduce memory usage
        max_length: Maximum sequence length (None uses model default)
    
    Caption Processing:
        shuffle_captions: Whether to shuffle comma-separated caption parts
        frozen_parts: Number of caption parts to keep fixed during shuffling
        caption_dropout: Probability of dropping caption parts (0.0-1.0)
        prompt: Task-specific prompt (e.g., "<MORE_DETAILED_CAPTION>" for Florence-2)
    
    Logging and Checkpointing:
        log_with: Logging backend ("wandb", "tensorboard", or None)
        name: Experiment name for logging and checkpoints
        seed: Random seed for reproducibility
        sample_every_n_epochs: Sample outputs every N epochs (None to disable)
        sample_every_n_steps: Sample outputs every N steps (None to disable)
        save_every_n_epochs: Save checkpoint every N epochs (None to disable)
        save_every_n_steps: Save checkpoint every N steps (None to disable)
    
    Example:
        ```python
        config = TrainingConfig(
            dropout=0.1,
            learning_rate=1e-4,
            batch_size=4,
            epochs=5,
            max_length=512,
            shuffle_captions=True,
            frozen_parts=1,
            caption_dropout=0.1,
            quantize=False,
            device="cuda",
            model_id="microsoft/Florence-2-base-ft",
            seed=42,
            log_with="wandb",
            name="my_experiment",
            prompt="<MORE_DETAILED_CAPTION>",
            gradient_checkpointing=True,
            gradient_accumulation_steps=2,
            sample_every_n_epochs=1,
            sample_every_n_steps=None,
            save_every_n_epochs=1,
            save_every_n_steps=None,
        )
        ```
    """
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    max_length: int | None
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
    sample_every_n_epochs: int | None
    sample_every_n_steps: int | None
    save_every_n_epochs: int | None
    save_every_n_steps: int | None


@dataclass
class PeftConfig:
    """Configuration for Parameter Efficient Fine-Tuning (PEFT) using LoRA.
    
    This class configures LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning
    of large vision-language models. LoRA adds trainable low-rank matrices to model layers
    while keeping the original weights frozen.
    
    LoRA Parameters:
        rank: LoRA rank/dimension (typically 4-64, higher = more parameters but better capacity)
        alpha: LoRA scaling factor (typically rank * 2, controls learning rate scaling)
        rslora: Whether to use RS-LoRA scaling (scales alpha by sqrt(rank))
        init_lora_weights: Weight initialization strategy
    
    Target Configuration:
        target_modules: Which model modules to apply LoRA to. Can be:
            - list[str]: Specific module names ["q_proj", "v_proj", "k_proj"]
            - str: Space-separated names "q_proj v_proj k_proj"  
            - None: Use model-specific defaults
    
    Weight Initialization Options:
        - "gaussian": Random Gaussian initialization (default, stable)
        - "eva": EVA initialization (can improve convergence)
        - "olora": Orthogonal LoRA initialization
        - "pissa": PISSA initialization method
        - "pissa_niter_[N]": PISSA with N iterations (e.g., "pissa_niter_4")
        - "loftq": LoFTQ initialization for quantized models
    
    Common Configurations by Model:
        Florence-2: rank=8, alpha=16, target_modules=["qkv", "proj", "fc1", "fc2"]
        BLIP: rank=8, alpha=16, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        GIT: rank=8, alpha=16, target_modules=["k_proj", "v_proj", "q_proj", "out_proj"]
    
    Memory and Performance Trade-offs:
        - Higher rank: More parameters, better capacity, more memory
        - Lower rank: Fewer parameters, faster training, less memory
        - More target_modules: Better coverage, more parameters
        - Fewer target_modules: Faster training, may limit adaptation
    
    Example:
        ```python
        # Basic LoRA config
        config = PeftConfig(
            rank=8,
            alpha=16,
            rslora=False,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            init_lora_weights="gaussian"
        )
        
        # High-capacity config for complex tasks
        config = PeftConfig(
            rank=32,
            alpha=64,
            rslora=True,
            target_modules=None,  # Use model defaults
            init_lora_weights="eva"
        )
        ```
    """
    rank: int
    alpha: int
    rslora: bool
    target_modules: list[str] | str | None
    init_lora_weights: Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"]


@dataclass
class OptimizerConfig:
    """Configuration for optimizer and advanced training techniques.
    
    This class configures the optimizer, learning rate scheduler, and advanced
    optimization techniques like low-rank optimization and LoRA+.
    
    Core Optimizer Settings:
        optimizer_name: Optimizer type ("AdamW", "DAdaptAdam", "Prodigy", etc.)
        optimizer_args: Additional optimizer arguments (e.g., {"weight_decay": 0.01})
        scheduler: Learning rate scheduler ("OneCycle", "linear", "cosine", None)
    
    Advanced Optimization (Optional):
        accumulation_rank: Rank for low-rank gradient accumulation (None to disable)
        activation_checkpointing: Whether to use activation checkpointing (None = auto)
        optimizer_rank: Rank for low-rank optimizer states (None to disable) 
        lora_plus_ratio: LoRA+ learning rate ratio for B matrices (None to disable)
    
    Common Optimizer Configurations:
        - AdamW: Most common, stable, good default choice
        - DAdaptAdam: Adaptive learning rate, requires no LR tuning
        - Prodigy: Advanced adaptive optimizer with better convergence
    
    Example:
        ```python
        # Basic AdamW config
        config = OptimizerConfig(
            optimizer_name="AdamW",
            optimizer_args={"weight_decay": 0.01, "betas": (0.9, 0.999)},
            scheduler="linear",
            accumulation_rank=None,
            activation_checkpointing=None,
            optimizer_rank=None,
            lora_plus_ratio=None,
        )
        
        # Advanced config with LoRA+ and low-rank optimization
        config = OptimizerConfig(
            optimizer_name="Prodigy",
            optimizer_args={"weight_decay": 0.01},
            scheduler="OneCycle", 
            accumulation_rank=8,
            activation_checkpointing=True,
            optimizer_rank=16,
            lora_plus_ratio=16.0,
        )
        ```
    """
    optimizer_args: dict
    optimizer_name: str
    scheduler: str
    accumulation_rank: int | None
    activation_checkpointing: bool | None
    optimizer_rank: int | None
    lora_plus_ratio: int | None


@dataclass
class LoraFileConfig:
    """Simple configuration for LoRA model output directory.
    
    Args:
        output_dir: Directory where trained LoRA weights will be saved
    """
    output_dir: Path


@dataclass
class FileConfig:
    """Configuration for dataset loading and file handling.
    
    This class configures how datasets are loaded, processed, and where files
    are read from. It supports both directory-based datasets (image/caption pairs)
    and single dataset files (JSONL, etc.).
    
    Dataset Sources (choose one):
        dataset_dir: Directory containing image/caption pairs (e.g., img.jpg + img.txt)
        dataset: Single dataset file (JSONL, parquet, etc.) or HuggingFace dataset name
    
    Caption File Configuration:
        combined_suffix: Suffix for combined caption files (e.g., "_combined")
        generated_suffix: Suffix for generated caption files (e.g., "_generated") 
        caption_file_suffix: Suffix for caption files (e.g., ".txt", default "")
    
    Processing Options:
        recursive: Whether to search subdirectories for image/caption pairs
        num_workers: Number of workers for data loading (0 = single-threaded)
        debug_dataset: Whether to print debug info about dataset loading
    
    Dataset Directory Structure Example:
        ```
        dataset_dir/
        ├── image1.jpg
        ├── image1.txt                    # Basic caption
        ├── image1_combined.txt           # Combined caption (if using combined_suffix)
        ├── image1_generated.txt          # Generated caption (if using generated_suffix)
        ├── image2.jpg
        ├── image2.txt
        └── subdirectory/                 # Searched if recursive=True
            ├── image3.jpg
            └── image3.txt
        ```
    
    Example:
        ```python
        # Directory-based dataset
        config = FileConfig(
            dataset_dir=Path("/path/to/images"),
            dataset=None,
            combined_suffix="_combined",
            generated_suffix="_generated", 
            caption_file_suffix="",
            recursive=True,
            num_workers=4,
            debug_dataset=False,
        )
        
        # Single dataset file
        config = FileConfig(
            dataset_dir=None,
            dataset=Path("/path/to/dataset.jsonl"),
            combined_suffix=None,
            generated_suffix=None,
            caption_file_suffix=None,
            recursive=False,
            num_workers=4,
            debug_dataset=False,
        )
        
        # HuggingFace dataset
        config = FileConfig(
            dataset_dir=None,
            dataset=Path("ybelkada/football-dataset"),  # HF dataset name
            combined_suffix=None,
            generated_suffix=None,
            caption_file_suffix=None,
            recursive=False,
            num_workers=0,
            debug_dataset=True,
        )
        ```
    """
    dataset_dir: Path | None
    dataset: Path | None
    combined_suffix: str | None
    generated_suffix: str | None
    caption_file_suffix: str | None
    recursive: bool
    num_workers: int
    debug_dataset: bool


@dataclass
class Trainer:
    """Main trainer class for vision-language model fine-tuning.

    Handles the training loop, batch processing, model saving, and logging
    for fine-tuning vision-language models like Florence-2 with LoRA.

    Attributes:
        model: PyTorch model to train (typically a vision-language model)
        processor: transformers.AutoProcessor for tokenizing and processing inputs
        optimizer: accelerate.AcceleratedOptimizer wrapping the base optimizer
        scheduler: Optional learning rate scheduler
        accelerator: accelerate.Accelerator for distributed training and mixed precision
        datasets: Datasets container with train_dataset and train_dataloader
        config: TrainingConfig with hyperparameters and training settings
        file_config: FileConfig with paths and dataset configuration
    """

    model: torch.nn.Module
    processor: AutoProcessor
    optimizer: AcceleratedOptimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    accelerator: Accelerator
    datasets: Datasets
    config: TrainingConfig
    file_config: FileConfig

    def process_batch(self, batch):
        """Process a single training batch with mixed precision.

        Args:
            batch: Dict with tensor values from dataloader:
                - "input_ids": torch.Tensor of shape (batch_size, seq_len)
                - "pixel_values": torch.Tensor of shape (batch_size, C, H, W)
                - "labels": torch.Tensor of shape (batch_size, seq_len)
                - "attention_mask": torch.Tensor of shape (batch_size, seq_len)

        Returns:
            Tuple of (model_outputs, loss) where:
            - model_outputs: Model output object with .loss attribute
            - loss: torch.Tensor scalar loss value for this batch
        """
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
                self.optimizer.zero_grad(set_to_none=True)
                with self.accelerator.accumulate(self.model):
                    _, loss = self.process_batch(batch)

                    self.accelerator.backward(loss)
                    self.optimizer.step()

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
                        self.sample(batch)

                    # Save
                    if self.config.save_every_n_steps is not None and step % self.config.save_every_n_steps == 0:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.save_model(f"step-{step}")

            # Sample
            if self.config.sample_every_n_epochs is not None and epoch % self.config.sample_every_n_epochs == 0:
                self.optimizer.eval()
                self.sample(batch)

            # Save
            if self.config.save_every_n_epochs is not None and epoch % self.config.save_every_n_epochs == 0:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.save_model(f"epoch-{epoch}")

        self.save_model()

    @torch.no_grad()
    def sample(self, batch: dict[str, torch.Tensor]):
        texts = []
        for item in batch["text"]:
            texts.append(item)
        del batch["text"]
        with self.accelerator.autocast():
            device = self.model.device
            dtype = self.model.dtype
            generated_output = self.model.generate(
                input_ids=batch["input_ids"].to(device),
                pixel_values=batch["pixel_values"].to(device, dtype=dtype),
                max_new_tokens=256,
            )

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
    arggroup.add_argument(
        "--init_lora_weights",
        type=str,
        default="gaussian",
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
