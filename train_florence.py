import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List
from accelerate import Accelerator
from tqdm import tqdm

import torch
import torch.optim as optim
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoProcessor

import flora_opt

from caption_train.captions import shuffle_caption


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
        "-ac",
        "--activation_checkpointing",
        action="store_true",
    )
    argparser.add_argument(
        "--accumulation_rank",
        type=int,
        default=None,
    )
    argparser.add_argument(
        "--optimizer_rank",
        type=int,
        default=None,
    )

    return argparser


class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        return self.loss_total / len(self.loss_list)


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
    ):
        self.dataset = dataset
        self.processor = processor
        self.shuffle_captions = shuffle_captions
        self.caption_dropout = caption_dropout
        self.frozen_parts = frozen_parts
        self.transform = transform
        self.task = "<DETAILED_CAPTION>"

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
            # padding="longest",
            # padding=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_token_type_ids=False,  # no need to set this to True since BART does not use token type ids
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
    dataset: Union[HFDataset, DatasetDict]
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


@dataclass
class Trainer:
    model: AutoModelForCausalLM
    processor: AutoProcessor
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
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

        for epoch in range(self.config.epochs):
            print("Epoch:", epoch)
            progress = tqdm(
                total=len(self.datasets.train_dataloader),
            )
            for idx, (batch, labels) in enumerate(
                self.datasets.train_dataloader
            ):
                step += 1
                with self.accelerator.accumulate(self.model):
                    outputs = self.process_batch(batch)

                    loss = outputs.loss

                    loss_recorder.add(
                        epoch=epoch,
                        step=idx,
                        loss=loss.detach().item(),
                    )

                    self.accelerator.backward(loss)

                    progress.set_postfix(
                        {
                            "loss": loss_recorder.moving_average,
                            # "lr": self.scheduler.get_last_lr(),
                        }
                    )
                    # print("Loss:", loss.item())

                    # loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    # self.scheduler.step()

                    progress.update()

                if idx % (len(self.datasets.train_dataloader) // 5) == 0:
                    self.sample(batch, labels)

        print(f"Saved to {self.config.output_dir}")
        # hugging face models saved to the directory
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
        )

    @torch.no_grad()
    def sample(self, batch, texts):
        with self.accelerator.autocast():
            generated_output = self.model.generate(
                **batch,
                max_new_tokens=75,
                do_sample=False,
                num_beams=3,
            )
            decoded = self.processor.batch_decode(
                generated_output, skip_special_tokens=True
            )

        # text = batch.pop("text")

        for gen, cap in zip(decoded, texts):
            print(f"Gen: {gen}")
            print(f"Cap: {cap}")


# Set the model as ready for training, makes sure the gradient are on


def set_up_model(model_id, training_config, device):
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
        use_rslora=True,
        init_lora_weights="gaussian",
    )

    # loading the model in float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

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

    train_dataset = ImageCaptioningDataset(dataset, processor)

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

    # Set batch_size to the batch that works for you.
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=training_config.batch_size,
        collate_fn=collate,
    )

    datasets = Datasets(dataset, train_dataset, train_dataloader)

    return datasets


# All the linear modules
BLIP_TARGET_MODULES = [
    "self.query",
    "self.key",
    "self.value",
    "output.dense",
    "self_attn.qkv",
    "self_attn.projection",
    "mlp.fc1",
    "mlp.fc2",
]

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
    )

    model, processor = set_up_model(
        args.model_id, training_config, args.device
    )

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    datasets = set_up_datasets(
        args.dataset_dir, processor, training_config, args.device
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    # accelerator = flora_opt.FloraAccelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     accumulation_compression_rank=args.accumulation_rank,
    # )

    model, processor = accelerator.prepare(model, processor)
    datasets.accelerate(accelerator)

    print(f"Setup optimizer LR={training_config.learning_rate}")

    # Setup AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # optimizer = flora_opt.Flora(
    #     model.parameters(),
    #     lr=training_config.learning_rate,
    #     rank=args.optimizer_rank,
    #     relative_step=False,
    # )

    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=training_config.learning_rate,
    #     # total_steps=None,
    #     epochs=training_config.epochs,
    #     cycle_momentum=False,
    #     steps_per_epoch=len(datasets.train_dataloader),
    #     base_momentum=0.85,
    #     max_momentum=0.95,
    # )
    #
    # optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    optimizer = accelerator.prepare(optimizer)

    trainer = Trainer(
        model=model,
        processor=processor,
        optimizer=optimizer,
        # scheduler=scheduler,
        scheduler=None,
        accelerator=accelerator,
        datasets=datasets,
        config=training_config,
    )
    trainer.train()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""
    Caption trainer for Florence

    Designed to be used with Hugging Face datasets.

    ---

    Use compile_captions.py to create a compatible dataset from
    image/text pairing.

    Example: a.png a.txt

    Creates a datasets compatible metadata.jsonl from those pairings.
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

    argparser = training_config_args(argparser)

    args = argparser.parse_args()
    main(args)
