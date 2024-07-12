import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model

# from peft import prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import BitsAndBytesConfig


from caption_train.captions import shuffle_caption
from caption_train.opt import (
    get_accelerator,
    get_optimizer,
    get_scheduler,
    opt_config_args,
)
from caption_train.util import LossRecorder
# from caption_train.util import print_gpu_utilization


def training_config_args(argparser):
    argparser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Rank/dim for the LoRA. Default: 16",
    )
    argparser.add_argument(
        "--alpha",
        type=float,
        default=32,
        help="Alpha for scaling the LoRA weights. Default: 32",
    )
    argparser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout for the LoRA network. Default: 0.05",
    )
    argparser.add_argument(
        "--target_modules",
        nargs="+",
        type=list,
        default=FLORENCE_TARGET_MODULES,
        help=f"Target modules to be trained. Consider Linear and Conv2D modules in the base model. Default: {' '.join(FLORENCE_TARGET_MODULES)}.",
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the LoRA. Default: 1e-4",
    )
    argparser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for the AdamW optimizer. Default: 1e-4",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the image/caption pairs. Default: 1",
    )
    argparser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to run. Default: 5",
    )
    argparser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Gradient checkpointing to reduce memory usage in exchange for slower training",
    )
    argparser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the training model to 4-bit",
    )
    argparser.add_argument(
        "--accumulation_compression",
        action="store_true",
        help="Accumulation compression for FloraAccelerator",
    )
    argparser.add_argument(
        "--rslora",
        action="store_true",
        help="RS LoRA scales alpha to size of rank",
    )
    argparser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        help="Sample the dataset every n epochs",
    )
    argparser.add_argument(
        "--sample_every_n_steps",
        type=int,
        help="Sample the dataset every n steps",
    )
    argparser.add_argument(
        "--save_every_n_epochs",
        type=int,
        help="Save the model every n epochs",
    )
    argparser.add_argument(
        "--save_every_n_steps",
        type=int,
        help="Save the model every n steps",
    )

    return argparser


IGNORE_ID = -100  # Pytorch ignore index when computing loss
MAX_LENGTH = 77


class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        dataset,
        processor,
        transform=None,
        caption_dropout=0.0,
        frozen_parts=0,
        shuffle_captions=False,
        task="<DETAILED_CAPTION>",
    ):
        self.dataset = dataset
        self.processor = processor
        self.shuffle_captions = shuffle_captions
        self.caption_dropout = caption_dropout
        self.frozen_parts = frozen_parts
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = (
            self.transform(item["image"])
            if self.transform is not None
            else item["image"]
        )

        encoding = self.processor(
            text=self.task,
            images=image,
            # padding="max_length",
            return_tensors="pt",
        )

        text = (
            shuffle_caption(
                item["text"],
                frozen_parts=self.frozen_parts,
                dropout=self.caption_dropout,
            )
            if self.shuffle_captions
            else item["text"]
        )
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            return_token_type_ids=False,
        )["input_ids"]

        labels[labels == self.processor.tokenizer.pad_token_id] = (
            IGNORE_ID  # do not learn to predict pad tokens during training
        )
        encoding["label"] = labels

        # # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding, text


@dataclass
class Datasets:
    dataset: DatasetDict
    train_dataset: ImageCaptioningDataset
    train_dataloader: DataLoader

    def accelerate(self, accelerator):
        self.train_dataloader = accelerator.prepare(self.train_dataloader)


@dataclass
class TrainingConfig:
    rank: int
    alpha: float
    dropout: float
    target_modules: List[str]
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    output_dir: str
    rslora: bool
    optimizer_rank: Optional[int]
    gradient_accumulation_steps: int
    sample_every_n_epochs: Optional[int]
    sample_every_n_steps: Optional[int]
    save_every_n_epochs: Optional[int]
    save_every_n_steps: Optional[int]


@dataclass
class Trainer:
    model: AutoModelForCausalLM
    processor: AutoProcessor
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    accelerator: Accelerator
    datasets: Datasets
    config: TrainingConfig

    def process_batch(self, batch):
        with self.accelerator.autocast():
            return self.model(**batch)

    def train(self):
        self.model.train()
        loss_recorder = LossRecorder()

        step = 0

        progress = tqdm(
            total=(
                len(self.datasets.train_dataloader)
                // self.config.gradient_accumulation_steps
            )
            * self.config.epochs,
        )
        for epoch in range(self.config.epochs):
            epoch = epoch + 1
            print("Epoch:", epoch)
            for idx, (batch, labels) in enumerate(
                self.datasets.train_dataloader
            ):
                with self.accelerator.accumulate(self.model):
                    outputs = self.process_batch(batch)

                    loss = outputs.loss

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
                        progress.set_postfix(
                            {"avg_loss": loss_recorder.moving_average}
                        )
                        progress.update()

                        self.accelerator.log(
                            {"avg_loss": loss_recorder.moving_average},
                            step=step,
                        )

                    # Sample
                    if (
                        self.config.sample_every_n_steps is not None
                        and step % self.config.sample_every_n_steps == 0
                    ):
                        self.sample(batch, labels)

                    # Save
                    if (
                        self.config.save_every_n_steps is not None
                        and step % self.config.save_every_n_steps == 0
                    ):
                        self.save_model(f"step-{step}")

            # Sample
            if (
                self.config.sample_every_n_epochs is not None
                and epoch % self.config.sample_every_n_epochs == 0
            ):
                self.sample(batch, labels)

            # Save
            if (
                self.config.save_every_n_epochs is not None
                and epoch % self.config.save_every_n_epochs == 0
            ):
                self.save_model(f"epoch-{epoch}")

        self.save_model()

    def sample(self, batch, texts):
        with torch.no_grad(), self.accelerator.autocast():
            generated_output = self.model.generate(
                **batch,
                max_new_tokens=75,
                do_sample=False,
                num_beams=3,
            )

        decoded = self.processor.batch_decode(
            generated_output, skip_special_tokens=True
        )

        for gen, cap in zip(decoded, texts):
            print(f"Gen: {gen}")
            print(f"Cap: {cap}")

    def save_model(self, name=None):
        # hugging face models saved to the directory
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if name is not None:
            name = "_" + str(name)
        else:
            name = ""
        unwrapped_model.save_pretrained(
            Path(self.config.output_dir + name),
            save_function=self.accelerator.save,
        )

        print(f"Saved to {self.config.output_dir + name}")


def set_up_model(model_id, training_config, args):
    quantization_config = None
    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    # model = prepare_model_for_kbit_training(
    #     model, use_gradient_checkpointing=True
    # )

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        print("Using gradient checkpointing")
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        print("Using gradient checkpointing")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    config = LoraConfig(
        r=training_config.rank,
        lora_alpha=training_config.alpha,
        lora_dropout=training_config.dropout,
        bias="none",
        # target specific modules (look for Linear in the model)
        # print(model) to see the architecture of the model
        target_modules=training_config.target_modules,
        task_type="CAUSAL_LM",
        inference_mode=False,
        use_rslora=training_config.rslora,
        init_lora_weights="gaussian",
        # init_lora_weights="olora",
    )

    # We layer our PEFT on top of our model using the PEFT config
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model, processor


def set_up_datasets(
    dataset_dir: Path,
    processor: AutoProcessor,
    training_config: TrainingConfig,
    device: str,
) -> Datasets:
    # We are extracting the train dataset
    dataset = load_dataset(
        "imagefolder",
        data_dir=dataset_dir,
        split="train",
    )

    print(len(dataset))
    print(next(iter(dataset)))

    train_dataset = ImageCaptioningDataset(
        dataset,
        processor,
    )

    def collate(items):
        # pad the input_ids and attention_mask
        input_ids = []
        attention_masks = []
        pixel_values = []
        labels = []
        texts = []
        for item, item_text in items:
            input_ids.append(item["input_ids"])
            attention_masks.append(item["attention_mask"])
            pixel_values.append(item["pixel_values"])
            labels.append(item["label"])
            texts.append(item_text)

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_masks),
            "pixel_values": torch.stack(pixel_values),
        }, texts

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator(),
        batch_size=training_config.batch_size,
        collate_fn=collate,
    )

    datasets = Datasets(dataset, train_dataset, train_dataloader)

    return datasets


# All the linear modules
FLORENCE_TARGET_MODULES = [
    "qkv",
    "proj",
    "k_proj",
    "v_proj",
    "q_proj",
    "out_proj",
    # "fc1",
    # "fc2",
]


def main(args):
    training_config = TrainingConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        optimizer_rank=args.optimizer_rank,
        rslora=args.rslora,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        sample_every_n_epochs=args.sample_every_n_epochs,
        sample_every_n_steps=args.sample_every_n_steps,
        save_every_n_epochs=args.save_every_n_epochs,
        save_every_n_steps=args.save_every_n_steps,
    )

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    print(f"Output results into: {str(Path(args.output_dir))}")

    model, processor = set_up_model(args.model_id, training_config, args)

    datasets = set_up_datasets(
        args.dataset_dir, processor, training_config, args.device
    )

    # Setup AdamW
    optimizer = get_optimizer(model, args.optimizer_name, training_config)
    scheduler = None

    if args.scheduler:
        scheduler = get_scheduler(
            optimizer,
            training_config,
            args,
            steps_per_epoch=len(datasets.train_dataloader),
        )

    accelerator = get_accelerator(args)
    model, processor, scheduler, optimizer = accelerator.prepare(
        model, processor, scheduler, optimizer
    )
    datasets.accelerate(accelerator)

    trainer = Trainer(
        model=model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        datasets=datasets,
        config=training_config,
    )
    trainer.train()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""
        Florence 2 trainer


    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    argparser.add_argument(
        "--dataset_dir",
        help="Dataset directory with the metadata.jsonl file and images",
    )
    argparser.add_argument(
        "--output_dir", help="Save the LoRA files to this directory"
    )
    argparser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the training on. Default: cuda or cpu",
    )
    argparser.add_argument(
        "--model_id",
        default="microsoft/Florence-2-base-ft",
        help="Model to train on. microsoft/Florence-2-base-ft or microsoft/Florence-2-large-ft. Default: microsoft/Florence-2-base-ft",
    )
    argparser.add_argument(
        "--seed", default=None, help="Seed used for random numbers"
    )
    argparser.add_argument(
        "--log_with",
        choices=["all", "wandb", "tensorboard"],
        help="Log with. all, wandb, tensorboard",
    )
    argparser.add_argument(
        "--name", help="Name to be used with saving and logging"
    )

    argparser = training_config_args(argparser)
    argparser = opt_config_args(argparser)

    args = argparser.parse_args()
    main(args)
