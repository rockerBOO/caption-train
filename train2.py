import argparse
import random
from pathlib import Path
from time import gmtime, strftime
import sys

import torch
import torchvision.transforms as T
from accelerate import Accelerator
from datasets import load_dataset

# from matplotlib import pyplot as plt
from peft import IA3Config, LoraConfig, get_peft_model
from prodigyopt import Prodigy
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
)
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    BlipForConditionalGeneration,
)

from caption_train.captions import setup_metadata, shuffle_caption


@torch.no_grad()
def sample(example, model, processor, accelerator):
    # with accelerator.autocast():
    #     inputs = processor(images=example["image_filename"], return_tensors="pt").to(
    #         accelerator.device
    #     )
    # pixel_values = inputs.pixel_values

    with accelerator.autocast():
        generated_ids = model.generate(
            pixel_values=example["pixel_values"],
            num_beams=3,
            min_length=3,
            max_length=75,
        )
        generated_captions = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    for example_caption in processor.batch_decode(
        example["input_ids"], skip_special_tokens=True
    ):
        print(f"Cap: {example_caption}")

    for generated_caption in generated_captions:
        print(f"Gen: {generated_caption}")


# def plot():
#     fig = plt.figure(figsize=(100, 100))
#
#     images = [ds["test"][x] for x in range(10)]
#
#     # Calculate the number of rows and columns based on the number of items in the data array
#     num_items = len(images)
#     num_rows = int(num_items**0.5)  # Calculate rows (square root for a balanced grid)
#     num_columns = (
#         num_items + num_rows - 1
#     ) // num_rows  # Calculate columns to fit images
#
#     # Create and configure the subplot grid
#     fig, axes = plt.subplots(
#         num_rows, num_columns, figsize=(320, 180)
#     )  # Adjust figsize as needed
#     fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing
#
#     with torch.no_grad():
#         # prepare image for the model
#         for i, example in enumerate(tqdm(images)):
#             image = example["image"]
#             inputs = processor(images=image, return_tensors="pt").to(device)
#             pixel_values = inputs.pixel_values
#
#             generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
#             generated_caption = processor.batch_decode(
#                 generated_ids, skip_special_tokens=True
#             )[0]
#
#             del inputs
#
#             row = i // num_columns
#             col = i % num_columns
#             ax = axes[row, col]
#             # fig.add_subplot(2, 3, i + 1)
#             ax.imshow(image)
#             ax.axis("off")
#             ax.set_title(f"Generated caption: {generated_caption}")
#
#     plt.savefig("x.png")
#


# def load_captions(dir: Path, captions=[]) -> list[str]:
#     for file in dir.iterdir():
#         if file.is_dir():
#             print(f"found dir: {file}")
#             captions = load_captions(file, captions)
#             continue
#
#         if file.suffix not in [".png", ".jpg", ".jpeg", ".webp"]:
#             continue
#
#         # need to check for images and then get the associated .txt file
#
#         txt_file = file.with_name(f"{file.stem}.txt")
#
#         if txt_file.exists():
#             with open(txt_file, "r") as f:
#                 file_name = str(file.relative_to(dataset_dir))
#                 caption = {
#                     "file_name": file_name,
#                     "text": " ".join(f.readlines()).strip(),
#                 }
#
#             captions.append(caption)
#
#         else:
#             print(f"no captions for {txt_file}")
#     return captions
#

# Convert .txt captions to metadata.jsonl file for dataset
# def setup_metadata(output, captions):
#     captions = load_captions(output, captions)
#
#     if len(captions) == 0:
#         raise ValueError("yo no captions")
#
#     # print(json.dumps(captions, indent=4))
#
#     print("Saving captions")
#
#     with open(Path(output) / "metadata.jsonl", "w") as f:
#         # jsonl has json for each item
#         for item in captions:
#             f.write(json.dumps(item) + "\n")
#

# def shuffle_caption(text, shuffle_on=", ", frozen_parts=1, dropout=0.0):
#     parts = [part.strip() for part in text.split(shuffle_on)]
#
#     # we want to keep the first part of the text, but shuffle the rest
#     frozen = []
#
#     if frozen_parts > 0:
#         for i in range(frozen_parts):
#             frozen.append(parts.pop(0))
#
#     final_parts = []
#
#     if dropout > 0.0:
#         final_parts = []
#         for part in parts:
#             rand = random.random()
#             if rand > dropout:
#                 final_parts.append(part)
#     else:
#         final_parts = parts
#
#     random.shuffle(final_parts)
#
#     return shuffle_on.join(frozen + final_parts)
#


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
        if self.transform is not None:
            # print("Transforming image...")
            # print(self.transform)
            # print(item["text"])
            image = self.transform(item["image"])
        else:
            image = item["image"]

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


def collate_fn(processor):
    def process_collate_fn(batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch],
                    padding=True,
                    return_tensors="pt",
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch

    return process_collate_fn


def setup_dataset(processor, args):
    setup_metadata(Path(args.dataset_dir), captions=[])

    ds = load_dataset(
        "imagefolder",
        # num_proc=4,
        data_dir=args.dataset_dir,
        split="train",
    )

    ds = ds.train_test_split(test_size=0.2, shuffle=True)

    # AUGMENTATIONS
    train_transforms = Compose(
        [
            RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.5, hue=0.3),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.RandomRotation(degrees=(-90, 90)),
            T.RandomAdjustSharpness(sharpness_factor=2),
            # T.RandomAutocontrast(),
        ]
    )

    train_dataset = ImageCaptioningDataset(
        ds["train"],
        processor,
        train_transforms,
        frozen_parts=args.frozen_parts,
        caption_dropout=args.caption_dropout,
        shuffle_captions=args.shuffle_captions,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=collate_fn(processor),
    )

    val_dataset = ImageCaptioningDataset(ds["test"], processor)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=collate_fn(processor),
    )

    print(ds["train"])

    print(ds["train"][0]["text"])
    print(ds["train"][0]["image"])

    return train_dataset, train_dataloader, val_dataset, val_dataloader


def wrap_in_ia3(model, args):
    print("Using IA3")
    ## IA3

    # for key in model.state_dict().keys():
    #     print(key)

    # qkv, projection, fc1, fc2, query, key, value, dense, decoder
    # target_modules = ["query", "value"]
    target_modules = (
        ".*encoder.*(self_attn|self|crossattention).*(qkv|key|value|dense|projection).*"
    )
    ff_modules = [
        "crossattention.output.dense",
        "attention.output.dense",
        "self_attn.projection",
    ]
    peft_config = IA3Config(
        target_modules=target_modules, feedforward_modules=ff_modules
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, peft_config


def wrap_in_lora(model, args):
    print("Using LoRA")
    ## LORA

    lora_rank = 8
    lora_alpha = 32
    lora_dropout = 0.05
    # qkv, projection, fc1, fc2, query, key, value, dense, decoder
    target_modules = [
        "query",
        "key",
        "qkv",
        "crossattention.output.dense",
        "attention.output.dense",
        "self_attn.projection",
    ]

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, peft_config


def get_scheduler(optimizer, train_dataset_length, args):
    ## SCHEDULER

    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, 2e-5, 2e-3, step_size_up=100, step_size_down=100, cycle_momentum=False
    # )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=int(len(train_dataset)/epochs), eta_min=0.5
    # )

    # scheduler = torch.ptim.lr_scheduler.ConstantLR(optimizer)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=1.2,
    #     steps_per_epoch=int(len(train_dataloader) / gradient_accumulation_steps),
    #     epochs=epochs,
    # )

    # CosineAnnealingLR

    n_epochs = 1
    steps = n_epochs * (
        train_dataset_length / (args.batch_size * args.gradient_accumulation_steps)
    )
    scheduler_args = {"steps": steps}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    # CosineAnnealingWarmRestarts

    n_epochs = 1
    steps = int(
        n_epochs
        * (train_dataset_length / (args.batch_size * args.gradient_accumulation_steps))
    )
    t_mult = 2
    scheduler_args = {"steps": steps, "t_mult": t_mult}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, steps, T_mult=t_mult
    )

    return scheduler, scheduler_args


def get_optimizer(model, args):
    # OPTIMIZERS

    # PRODIGY

    lr = 1.0
    weight_decay = 0.1
    optimizer_args = {
        "lr": lr,
        "weight_decay": weight_decay,
        # "safeguard_warmup": True,
        "use_bias_correction": True,
        "d_coef": 1.0,
    }
    # optimizer_args = {"lr": lr, }
    print("Using Prodigy optimizer")
    print(optimizer_args)
    optimizer = Prodigy(model.parameters(), **optimizer_args)

    # DADAPTATION

    # from dadaptation import DAdaptAdam

    # lr = 1.0
    # weight_decay = 0.05
    # optimizer_args = {"lr": lr, "decouple": True}
    # print("Using DAdaptation optimizer")
    # print(optimizer_args)
    # optimizer = DAdaptAdam(model.parameters(), weight_decay=weight_decay,
    #   **optimizer_args)

    # ADAM
    # lr = 2e-3
    # weight_decay = 1e-4
    # optimizer_args = {"lr": lr, "weight_decay": weight_decay}
    # optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)

    return optimizer, optimizer_args


def get_blip_model(args):
    blip_config = AutoConfig.from_pretrained(args.model_name_or_path)

    # TODO: map model_config to blip_config
    blip_config.hidden_dropout_prob = 0.2
    blip_config.attention_probs_dropout_prob = 0.3

    model = BlipForConditionalGeneration.from_pretrained(
        # "Salesforce/blip-image-captioning-base"
        args.model_name_or_path,
        device_map={"": 0},
        config=blip_config,
    )

    return model


def train(model, processor, train_dataloader, val_dataloader, args):
    # epochs = 5
    # save_every_n_epochs = 5
    # gradient_accumulation_steps = 2
    # gradient_checkpointing = True
    # max_grad_norm = 1.0

    # dataset
    # batch_size = 8
    # shuffle_captions = True
    # frozen_parts = 1
    # caption_dropout = 0.1

    print(model)

    # PEFT MODULE
    peft_module = "LoRA"
    print(peft_module == "LoRA", peft_module)

    if peft_module == "LoRA":
        model, peft_config = wrap_in_lora(model, args)
    elif peft_module == "IA3":
        model, peft_config = wrap_in_ia3(model, args)
    else:
        raise ValueError(f"Invalid PEFT module: {peft_module}")

    # OPTIMIZER
    optimizer, optimizer_args = get_optimizer(model, args)

    # SCHEDULER
    scheduler, scheduler_args = get_scheduler(
        optimizer, len(train_dataloader.dataset), args
    )

    # OUTPUT
    output_dir = Path(args.output_dir)

    training_datetime = strftime("%Y-%m-%d-%H%M%S", args.training_start)
    save_to = output_dir / f"{args.training_name}-{training_datetime}"

    # ACCELERATOR
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
    )

    def clean_dict(unclean_dict):
        cleaned_dict = {}

        for key in unclean_dict.keys():
            v = unclean_dict.get(key)
            print(type(v))

            if type(v) not in [bool, str, int, float, dict]:
                continue

            cleaned_dict[key] = v

        return cleaned_dict

    project_config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optimizer": type(optimizer).__name__,
        "optimizer_args": optimizer_args,
        "scheduler": type(scheduler).__name__,
        "scheduler_args": scheduler_args,
        "peft": type(peft_config).__name__,
        "peft_config": {**clean_dict(peft_config.__dict__)},
    }

    print(project_config)

    accelerator.init_trackers(project_name=args.training_name, config=project_config)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # TRAIN
    model.train()

    # Should we compile here? lets see
    # model = torch.compile(model, backend="inductor")

    global_step = 0

    loss_list = []
    loss_total = 0.0

    val_loss_list = []
    val_loss_total = 0.0

    # EPOCHS
    for epoch in range(args.epochs):
        print("Epoch:", epoch + 1)
        # for idx, batch in enumerate(train_dataloader):
        t_dataloader = tqdm(train_dataloader)

        # BATCH
        for step, batch in enumerate(t_dataloader):
            with accelerator.accumulate(model):
                global_step += 1

                input_ids = batch.pop("input_ids")
                pixel_values = batch.pop("pixel_values")
                attention_mask = batch.pop("attention_mask")

                with accelerator.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )

                loss = outputs.loss.mean()

                accelerator.backward(loss)

                if (
                    accelerator.sync_gradients
                    and args.max_grad_norm is not None
                    and args.max_grad_norm != 0.0
                ):
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                current_loss = loss.detach().item()
                if epoch == 0:
                    loss_list.append(current_loss)
                else:
                    loss_total -= loss_list[step]
                    loss_list[step] = current_loss

                loss_total += current_loss
                avg_loss = loss_total / len(loss_list)

                loggable = {
                    "loss/avg": avg_loss,
                    "loss/current": current_loss,
                    "lr/lr": scheduler.get_last_lr()[0],
                }

                if "d" in optimizer.param_groups[0]:
                    loggable["lr/d*lr"] = (
                        optimizer.param_groups[0].get("d") * scheduler.get_last_lr()[0]
                    )

                accelerator.log(loggable, step=global_step)
                t_dataloader.set_postfix(
                    {
                        "loss": avg_loss,
                        # "lr": scheduler.get_last_lr()[0],
                        # "d*lr": optimizer.param_groups[0].get("d")
                        # * scheduler.get_last_lr()[0],
                    }
                )

        # VALIDATION
        v_dataloader = val_dataloader
        with torch.no_grad():
            for val_step, batch in enumerate(v_dataloader):
                input_ids = batch.pop("input_ids").to(accelerator.device)
                pixel_values = batch.pop("pixel_values").to(accelerator.device)
                attention_mask = batch.pop("attention_mask").to(accelerator.device)

                with accelerator.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )

                loss = outputs.loss.mean()

                current_loss = loss.detach().item()

                if epoch == 0:
                    val_loss_list.append(current_loss)
                else:
                    val_loss_total -= val_loss_list[val_step]
                    val_loss_list[val_step] = current_loss

                val_loss_total += current_loss

        avg_loss = val_loss_total / len(val_loss_list)
        print(f"Validation avg loss {avg_loss}")

        loggable = {"loss/val": avg_loss}

        accelerator.log(loggable, step=global_step)

        # load image
        for val in val_dataloader:
            sample(val, model, processor, accelerator)

        # Save every n epochs
        if (
            args.save_every_n_epochs != 0
            and args.save_every_n_epochs % (epoch + 1) == 0
        ):
            accelerator.wait_for_everyone()
            save_epoch_to = f"{save_to}_{epoch+1}"
            print(f"Saved to {save_epoch_to}")
            model.save_pretrained(save_epoch_to, safe_serialization=True)

    accelerator.end_training()

    # SAVE MODEL
    if (
        args.save_every_n_epochs != 0
        and args.save_every_n_epochs % (args.epochs + 1) != 0
    ):
        accelerator.wait_for_everyone()
        print(f"Saved to {save_to}")
        model.save_pretrained(save_to, safe_serialization=True)


def main(args):
    print(args)
    seed = args.seed or 42
    torch.manual_seed(seed)
    random.seed(seed)

    # args.dataset_dir = "/mnt/900/input/nsfw/captions"
    # args.training_name = "pov"

    # datetime = strftime("%Y-%m-%d-%H%M%S", gmtime())
    # args.save_to = f"training/sets/{args.training_name}-{datetime}"

    args.training_start = gmtime()

    model = get_blip_model(args)
    model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
    )

    # DATASET
    train_dataset, train_dataloader, val_dataset, val_dataloader = setup_dataset(
        processor, args
    )

    train(model, processor, train_dataloader, val_dataloader, args)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    # For example: training/sets/
    argparser.add_argument(
        "--output_dir", required=True, help="Directory to output the resulting model to"
    )

    argparser.add_argument(
        "--dataset_dir",
        required=True,
        help="Directory for where the image/captions are stored. Is recursive.",
    )

    argparser.add_argument("--lr", help="Learning rate for the training")

    argparser.add_argument(
        "--model_name_or_path",
        default="Salesforce/blip-image-captioning-large",
        help="Model name on hugging face or path to model. (Should be a BLIP model at the moment)",
    )

    argparser.add_argument(
        "--training_name",
        help="Training name used in creating the output directory and in logging (wandb)",
    )

    argparser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs to run the training for",
    )

    argparser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=0,
        help="Save an output of the current epoch every n epochs",
    )

    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of image/text pairs to train in the same batch",
    )
    argparser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    argparser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing",
    )

    argparser.add_argument(
        "--max_grad_norm", type=float, default=None, help="Max gradient norm"
    )

    argparser.add_argument(
        "--shuffle_captions", action="store_true", help="Shuffle captions when training"
    )

    argparser.add_argument(
        "--frozen_parts",
        default=0,
        help="How many parts (parts separated by ',') do we want to keep in place when shuffling",
    )

    argparser.add_argument(
        "--caption_dropout",
        default=0.0,
        help="Amount of parts we dropout in the caption.",
    )

    argparser.add_argument(
        "--peft_module",
        default="LoRA",
        choices=["LoRA", "IA3"],
        help="PEFT module to use in training",
    )

    argparser.add_argument(
        "--seed", default=42, help="Seed for the random number generation"
    )

    argparser.add_argument(
        "--interactive",
        default=False,
        help="Interactive hook for modifying the dataset while running",
    )

    # For example BLIP config for
    #   hidden_dropout_prob,
    #   attention_probs_dropout_prob,
    #   device_map={"*": 1}
    argparser.add_argument(
        "--model_config",
        help="Model configuration parameters (dropout, device_map, ...)",
    )

    # epochs = 5
    # save_every_n_epochs = 5
    # gradient_accumulation_steps = 2
    # gradient_checkpointing = True
    # max_grad_norm = 1.0
    #
    # # dataset
    # batch_size = 8
    # shuffle_captions = True
    # frozen_parts = 1
    # caption_dropout = 0.1
    #
    # peft_module = "LoRA"

    args = argparser.parse_args()
    main(args)
