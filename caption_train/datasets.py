import argparse
from pathlib import Path
from typing import Optional
import torch

from transformers import AutoProcessor
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

from caption_train.captions import shuffle_caption


class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        dataset,
        processor,
        transform=None,
        max_length: Optional[int] = 77,
        caption_dropout=0.0,
        frozen_parts=0,
        shuffle_captions=False,
        task="<MORE_DETAILED_CAPTION>",
    ):
        self.dataset = dataset
        self.processor = processor
        self.shuffle_captions = shuffle_captions
        self.max_length = max_length
        self.caption_dropout = caption_dropout
        self.frozen_parts = frozen_parts
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        item = self.dataset[idx]
        images = self.transform(item["image"]) if self.transform is not None else item["image"]

        encoding = self.processor(
            text=self.task,
            images=images,
            padding="max_length",
            max_length=self.max_length,
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
        labels = self.processor.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.max_length)

        encoding["label"] = labels
        #
        # # # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding, text


class ImageCaptionPairDataset(Dataset):
    def __init__(
        self,
        dataset,
        processor,
        transform=None,
        max_length: Optional[int] = None,
        caption_dropout=0.0,
        frozen_parts=0,
        shuffle_captions=False,
        task="<MORE_DETAILED_CAPTION>",
    ):
        self.dataset = dataset
        self.processor = processor
        self.shuffle_captions = shuffle_captions
        self.max_length = max_length
        self.caption_dropout = caption_dropout
        self.frozen_parts = frozen_parts
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(Image.open(item["image"])) if self.transform is not None else Image.open(item["image"])

        encoding = self.processor(
            text=self.task,
            images=image,
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        # num_sentences = max(
        #     [(len(t), len(g)) for t, g in zip(item["text"].split("."), item["generated_caption"].split("."))]
        # )
        # text = blend_sentences(item["text"], item["generated_caption"], num_sentences)

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
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        encoding["attention_mask"] = labels.attention_mask
        encoding["labels"] = labels.input_ids

        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding, text


@dataclass
class Datasets:
    train_dataset: Dataset
    train_dataloader: DataLoader

    def accelerate(self, accelerator):
        self.train_dataloader = accelerator.prepare(self.train_dataloader)


def set_up_image_text_pair(
    dataset_dir: Path, model: torch.nn.Module, processor: AutoProcessor, accelerator, training_config
) -> Datasets:
    images = (
        list(Path(dataset_dir).rglob("*.png"))
        + list(Path(dataset_dir).rglob("*.jpg"))
        + list(Path(dataset_dir).rglob("*.jpeg"))
        + list(Path(dataset_dir).rglob("*.webp"))
    )

    image_captions = []
    for image in images:
        image = Path(image)
        caption_file = image.with_name(image.stem + "_combined.txt")
        with open(caption_file, "r") as f:
            caption = f.read()
        image_captions.append(caption)

    # accelerator.print("Caching dataset")
    # pre_generated_captions: list[str] = []
    # for image in tqdm(images):
    #     processed: dict[str, torch.Tensor] = processor(
    #         images=[Image.open(image)], text=["<MORE_DETAILED_CAPTION>"], return_tensors="pt"
    #     )
    #     with torch.no_grad(), accelerator.autocast():
    #         generated = model.generate(**processed.to(model.device))
    #     generated = processor.batch_decode(generated, skip_special_tokens=True)
    #     pre_generated_captions.append(generated)
    # train_dataset = [
    #     {"image": image, "text": caption, "generated_caption": pre_generated_caption}
    #     for image, caption, pre_generated_caption in zip(images, image_captions, pre_generated_captions)
    # ]

    train_dataset = [{"image": image, "text": caption} for image, caption in zip(images, image_captions)]

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
            labels.append(item["labels"])
            texts.append(item_text)

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_masks),
            "pixel_values": torch.stack(pixel_values),
        }, texts

    image_pair_dataset = ImageCaptionPairDataset(
        train_dataset,
        processor,
        max_length=training_config.max_length,
        shuffle_captions=training_config.shuffle_captions,
        task=training_config.prompt,
    )

    train_dataloader = DataLoader(
        image_pair_dataset,
        shuffle=True,
        generator=torch.Generator(),
        batch_size=training_config.batch_size,
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
    )

    print(next(iter(train_dataloader)))

    datasets = Datasets(
        image_pair_dataset,
        train_dataloader,
    )

    return datasets


def set_up_datasets(dataset_dir: Path, processor: AutoProcessor, training_config, dataset_config) -> Datasets:
    # We are extracting the train dataset
    dataset = load_dataset(
        "imagefolder",
        data_dir=str(dataset_config.dataset_dir),
        split="train",
    )

    print(len(dataset))
    print(next(iter(dataset)))

    train_dataset = ImageCaptioningDataset(
        dataset,
        processor,
        max_length=training_config.max_length,
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

    datasets = Datasets(train_dataset, train_dataloader)

    return datasets


def datasets_config_args(argparser: argparse.ArgumentParser):
    arggroup = argparser.add_argument_group("Datasets")
    arggroup.add_argument(
        "--dataset_dir",
        type=Path,
        help="Dataset directory image and caption file pairs. img.jpg and img.txt",
    )
    arggroup.add_argument(
        "--dataset",
        type=Path,
        help="Dataset with the metadata.jsonl file and images",
    )
    arggroup.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Save the LoRA files to this directory",
    )

    return argparser, arggroup
