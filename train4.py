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
from transformers import AutoModelForVision2Seq, AutoProcessor

from caption_train.captions import shuffle_caption


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
            images=image,
            padding="max_length",
            return_tensors="pt",
        )
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = (
            shuffle_caption(
                item["text"],
                frozen_parts=self.frozen_parts,
                dropout=self.caption_dropout,
            )
            if self.shuffle_captions
            else item["text"]
        )
        # print("encoding")
        # print(encoding["text"])
        return encoding


def collator(processor, device):
    def collate(batch):
        print("COLLATING")
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack(
                    [example[key] for example in batch]
                )
                processed_batch[key].to(device)
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch],
                    padding=True,
                    return_tensors="pt",
                )
                processed_batch["text"] = [
                    example["text"] for example in batch
                ]
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs[
                    "attention_mask"
                ]
        print("processed", processed_batch.keys())
        return processed_batch

    return collate


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
        default=BLIP_TARGET_MODULES,
        help=f"Target modules to be trained. Consider Linear and Conv2D modules in the base model. Default: {' '.join(BLIP_TARGET_MODULES)}",
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the LoRA. Default: 1e-3",
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
        default=2,
        help="Batch size for the image/caption pairs. Default: 2",
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to run. Default: 5",
    )

    return argparser


@dataclass
class Trainer:
    model: AutoModelForVision2Seq
    processor: AutoProcessor
    optimizer: torch.optim.Optimizer
    accelerator: Accelerator
    datasets: Datasets
    config: TrainingConfig

    def process_batch(self, batch):
        input_ids = batch.get("input_ids").to(self.model.device)
        pixel_values = batch.get("pixel_values").to(self.model.device)
        attention_mask = batch.get("attention_mask").to(self.model.device)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        return outputs

    def train(self):
        loss_recorder = LossRecorder()

        step = 0

        for epoch in range(self.config.epochs):
            print("Epoch:", epoch)
            progress = tqdm(
                enumerate(self.datasets.train_dataloader),
                total=len(self.datasets.train_dataloader),
            )
            for idx, batch in progress:
                step += 1
                self.optimizer.zero_grad()

                outputs = self.process_batch(batch)

                loss = outputs.loss

                loss_recorder.add(
                    epoch=epoch,
                    step=idx,
                    loss=loss.item(),
                )

                progress.set_postfix({"loss": loss_recorder.moving_average})
                # print("Loss:", loss.item())

                loss.backward()

                self.optimizer.step()

                if idx % (len(self.datasets.train_dataloader) // 5) == 0:
                    generated_output = self.model.generate(
                        pixel_values=batch.pop("pixel_values").to(
                            self.model.device
                        ),
                        max_new_tokens=64,
                    )
                    decoded = self.processor.batch_decode(
                        generated_output, skip_special_tokens=True
                    )

                    text = batch.pop("text")

                    for gen, cap in zip(decoded, text):
                        print(f"Gen: {gen}")
                        print(f"Cap: {cap}")

        print(f"Saved to {self.config.output_dir}")
        # hugging face models saved to the directory
        self.model.save_pretrained(args.output_dir)


# Set the model as ready for training, makes sure the gradient are on


def setup_model(model_id, training_config, device):
    config = LoraConfig(
        r=training_config.rank,
        lora_alpha=training_config.alpha,
        lora_dropout=training_config.dropout,
        bias="none",
        # target specific modules (look for Linear in the model)
        # print(model) to see the architecture of the model
        target_modules=training_config.target_modules,
    )

    # loading the model in float16
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    model.to(device)
    print(model)
    processor = AutoProcessor.from_pretrained(model_id)

    # We layer our PEFT on top of our model using the PEFT config
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    model.train()

    return model, processor


def setup_datasets(
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

    def collate(batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack(
                    [example[key] for example in batch]
                )
                processed_batch[key].to(device)
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch],
                    padding=True,
                    return_tensors="pt",
                )
                processed_batch["text"] = [
                    example["text"] for example in batch
                ]
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs[
                    "attention_mask"
                ]
        return processed_batch

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

    model, processor = setup_model(args.model_id, training_config, args.device)

    datasets = setup_datasets(
        args.dataset_dir, processor, training_config, args.device
    )

    accelerator = Accelerator()

    model, processor = accelerator.prepare(model, processor)
    datasets.accelerate(accelerator)

    # Setup AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    trainer = Trainer(
        model=model,
        processor=processor,
        optimizer=optimizer,
        accelerator=accelerator,
        datasets=datasets,
        config=training_config,
    )
    trainer.train()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""
    Caption trainer for BLIP

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
        "dataset_dir",
        help="Dataset directory with the metadata.jsonl file and images",
    )
    argparser.add_argument(
        "output_dir", help="Save the LoRA files to this directory"
    )
    argparser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the training on. Default: cuda or cpu",
    )
    argparser.add_argument(
        "--model_id",
        default="Salesforce/blip-image-captioning-base",
        help="Model to train on. Salesforce/blip-image-captioning-base or Salesforce/blip-image-captioning-large. Default: Salesforce/blip-image-captioning-base",
    )

    argparser = training_config_args(argparser)

    args = argparser.parse_args()
    main(args)
