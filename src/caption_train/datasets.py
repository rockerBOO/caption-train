import argparse
from pathlib import Path
from collections.abc import Callable
import torch

from transformers import AutoProcessor
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

from caption_train.captions import shuffle_caption


class ImageCaptioningDataset(Dataset):
    """Dataset for image captioning tasks using pre-processed image tensors.

    This dataset expects items with pre-processed image tensors (not file paths).
    The processor is used to encode the task text and combine it with image data.

    Args:
        dataset: list of dicts with keys:
            - "image": torch.Tensor of shape (C, H, W) - pre-processed image tensor
            - "text": str - caption text for the image
        processor: transformers.AutoProcessor - processor for encoding text and images.
            Must support calling processor(text=..., images=..., return_tensors="pt").
            Returns dict with tensor values:
            - "input_ids": torch.Tensor of shape (1, seq_len)
            - "attention_mask": torch.Tensor of shape (1, seq_len)
            - "pixel_values": torch.Tensor of shape (1, C, H, W)
        transform: Callable to transform image tensors or None
        max_length: Maximum sequence length for text encoding
        caption_dropout: Fraction of caption parts to randomly drop (0.0-1.0)
        frozen_parts: Number of caption parts to keep fixed when shuffling
        shuffle_captions: Whether to shuffle caption parts during training
        task: Task prompt text sent to the processor (e.g., "<MORE_DETAILED_CAPTION>")

    Returns:
        Tuple of (encoding, text) where:
        - encoding: Dict with tensor values (batch dimension squeezed):
            * "input_ids": torch.Tensor of shape (seq_len,)
            * "attention_mask": torch.Tensor of shape (seq_len,)
            * "pixel_values": torch.Tensor of shape (C, H, W)
            * "label": Tokenizer output object with .input_ids and .attention_mask attributes
        - text: str - the processed caption text

    Note:
        The returned encoding has tensors with batch dimension squeezed out.
        The "label" key contains the tokenizer output object with input_ids and attention_mask.
    """

    def __init__(
        self,
        dataset: list[dict[str, torch.Tensor | str]],
        processor,  # transformers.AutoProcessor
        transform: Callable | None = None,
        max_length: int | None = 77,
        caption_dropout: float = 0.0,
        frozen_parts: int = 0,
        shuffle_captions: bool = False,
        task: str = "<MORE_DETAILED_CAPTION>",
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


class FlorenceImageTextDataset(Dataset):
    """Dataset for Florence-2 vision-language model training using image file paths.

    This dataset loads images from file paths using PIL and processes them with
    a Florence-2 processor. Different from ImageCaptioningDataset, this loads
    images from disk rather than using pre-processed tensors.

    Args:
        dataset: list of dicts with keys:
            - "image": str or Path - path to image file on disk
            - "text": str - caption text for the image
        processor: transformers.AutoProcessor - Florence-2 processor for encoding text and images.
            Must support calling processor(text=..., images=..., return_tensors="pt").
            Returns dict with tensor values:
            - "input_ids": torch.Tensor of shape (1, seq_len)
            - "attention_mask": torch.Tensor of shape (1, seq_len)
            - "pixel_values": torch.Tensor of shape (1, C, H, W)
            Tokenizer must support processor.tokenizer(text, return_tensors="pt", padding=...).
            Returns object with attributes:
            - .input_ids: torch.Tensor of shape (1, seq_len)
            - .attention_mask: torch.Tensor of shape (1, seq_len)
        transform: Callable to transform PIL Image objects or None
        max_length: Maximum sequence length for text encoding (defaults to processor max) or None
        caption_dropout: Fraction of caption parts to randomly drop (0.0-1.0)
        frozen_parts: Number of caption parts to keep fixed when shuffling
        shuffle_captions: Whether to shuffle caption parts during training
        task: Task prompt text sent to the processor (e.g., "<MORE_DETAILED_CAPTION>")

    Returns:
        Tuple of (encoding, text) where:
        - encoding: Dict with tensor values (batch dimension squeezed):
            * "input_ids": torch.Tensor of shape (seq_len,)
            * "attention_mask": torch.Tensor of shape (seq_len,)
            * "pixel_values": torch.Tensor of shape (C, H, W)
            * "labels": torch.Tensor of shape (seq_len,) - input_ids for loss calculation
        - text: str - the processed caption text

    Note:
        The returned encoding has tensors with batch dimension squeezed out.
        The "labels" key contains input_ids from tokenizer output for training loss.
        Images are loaded using PIL.Image.open() and can be transformed before processing.
    """

    def __init__(
        self,
        dataset: list[dict[str, str | Path]],
        processor,  # transformers.AutoProcessor for Florence-2
        transform: Callable | None = None,
        max_length: int | None = None,
        caption_dropout: float = 0.0,
        frozen_parts: int = 0,
        shuffle_captions: bool = False,
        task: str = "<MORE_DETAILED_CAPTION>",
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
    """Container for training datasets and dataloaders.

    Simple dataclass that holds both the dataset and dataloader for training,
    with a method to prepare them for distributed training with Accelerate.

    Attributes:
        train_dataset: The training dataset (ImageCaptioningDataset or FlorenceImageTextDataset)
        train_dataloader: PyTorch DataLoader wrapping the train_dataset
    """

    train_dataset: Dataset
    train_dataloader: DataLoader

    def accelerate(self, accelerator):
        """Prepare the dataloader for distributed training using Accelerate.

        Args:
            accelerator: accelerate.Accelerator instance for distributed training
        """
        self.train_dataloader = accelerator.prepare(self.train_dataloader)


def glob(path: Path, recursive=False):
    if recursive:
        return path.rglob("*.*")
    else:
        return path.glob("*.*")


def find_images(dir: Path, recursive=False) -> set[Path]:
    # Find images
    extensions = [".png", ".jpg", ".jpeg", ".webp"]
    images = {file for ext in extensions for file in glob(dir, recursive) if file.suffix.lower() in extensions}

    return images


def set_up_image_text_pair(
    model: torch.nn.Module, processor: AutoProcessor, accelerator, training_config, dataset_config
) -> Datasets:
    dataset_dir = dataset_config.dataset_dir

    images = find_images(dataset_dir, dataset_config.recursive)

    assert len(images) > 0, "No images found in the dataset"

    caption_file_suffix = dataset_config.caption_file_suffix
    if caption_file_suffix is None:
        caption_file_suffix = ""

    # Load captions for images
    image_captions = []
    for image in images:
        image = Path(image)
        caption_file = image.with_name(image.stem + caption_file_suffix + ".txt")

        if not Path(caption_file).exists():
            print(f"No caption foud for {image}")
            continue

        with open(caption_file, "r") as f:
            caption = f.read()
        image_captions.append(caption)

    assert len(image_captions) == len(images), "Did not find a caption for all images"
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
            "text": texts,
        }

    image_pair_dataset = FlorenceImageTextDataset(
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
        num_workers=dataset_config.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    if dataset_config.debug_dataset:
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
        }

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator(),
        batch_size=training_config.batch_size,
        collate_fn=collate,
    )

    datasets = Datasets(train_dataset, train_dataloader)

    return datasets


def lora_config_args(argparser: argparse.ArgumentParser):
    arggroup = argparser.add_argument_group("Lora Config")
    arggroup.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Save the LoRA files to this directory",
    )

    return argparser, arggroup


def datasets_config_args(argparser: argparse.ArgumentParser):
    arggroup = argparser.add_argument_group("Datasets")
    arggroup.add_argument(
        "--dataset_dir",
        type=Path,
        help="Dataset directory image and caption file pairs. img.jpg and img.txt",
    )
    arggroup.add_argument(
        "--combined_suffix",
        type=str,
        default=None,
        help="Suffix for the combined caption file. Default to `_combined`",
    )
    arggroup.add_argument(
        "--generated_suffix",
        type=str,
        default=None,
        help="Suffix for the generated caption file. Default to `_generated`",
    )
    arggroup.add_argument(
        "--caption_file_suffix",
        type=str,
        default=None,
        help='Suffix for the caption file. Default to ""',
    )
    arggroup.add_argument(
        "--recursive",
        action="store_true",
        help="Whether to recursively search for images and captions",
    )
    arggroup.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for the dataloader. Default to 0",
    )
    arggroup.add_argument(
        "--debug_dataset",
        action="store_true",
        help="Whether to print the first batch of the dataloader",
    )

    return argparser, arggroup
