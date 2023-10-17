from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    AutoConfig,
)

from prodigyopt import Prodigy

from torchvision.transforms import (
    # CenterCrop,
    Compose,
    # Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    # Resize,
    # ToTensor,
)
import torchvision.transforms as T

from peft import LoraConfig, IA3Config, get_peft_model

# from peft.tuners.lora import mark_only_lora_as_trainable
import torch
from pathlib import Path
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator
from time import gmtime, strftime
import gc
import random

dataset_dir = "/mnt/900/input/nsfw/captions"
# dataset_dir = (
#     "/mnt/900/input/others/knface_v9/base_knface_lr_balanced_renamed_prefixed/"
# )

training_name = "pov"

datetime = strftime("%Y-%m-%d-%H%M%S", gmtime())
save_to = f"training/sets/{training_name}-{datetime}"

# check if saveto directory is there


@torch.no_grad()
def sample(model, processor):
    # load image
    for i in range(10):
        example = ds["test"][i]
        image = example["image"]

        # prepare image for the model
        inputs = processor(images=image, return_tensors="pt").to(
            device, dtype=torch.float16
        )
        pixel_values = inputs.pixel_values

        generated_ids = model.generate(
            pixel_values=pixel_values, num_beams=3, min_length=3, max_length=75
        )
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        print(f"Cap: {example['text']}")
        print(f"Gen: {generated_caption}")


def plot():
    fig = plt.figure(figsize=(100, 100))

    images = [ds["test"][x] for x in range(10)]

    # Calculate the number of rows and columns based on the number of items in the data array
    num_items = len(images)
    num_rows = int(num_items**0.5)  # Calculate rows (square root for a balanced grid)
    num_columns = (
        num_items + num_rows - 1
    ) // num_rows  # Calculate columns to fit images

    # Create and configure the subplot grid
    fig, axes = plt.subplots(
        num_rows, num_columns, figsize=(320, 180)
    )  # Adjust figsize as needed
    fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing

    with torch.no_grad():
        # prepare image for the model
        for i, example in enumerate(tqdm(images)):
            image = example["image"]
            inputs = processor(images=image, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values

            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            del inputs

            row = i // num_columns
            col = i % num_columns
            ax = axes[row, col]
            # fig.add_subplot(2, 3, i + 1)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"Generated caption: {generated_caption}")

    plt.savefig("x.png")


def load_captions(dir: Path, captions=[]) -> list[str]:
    for file in dir.iterdir():
        if file.is_dir():
            print(f"found dir: {file}")
            captions = load_captions(file, captions)
            continue

        if file.suffix not in [".png", ".jpg", ".jpeg", ".webp"]:
            continue

        # need to check for images and then get the associated .txt file

        txt_file = file.with_name(f"{file.stem}.txt")

        if txt_file.exists():
            with open(txt_file, "r") as f:
                file_name = str(file.relative_to(dataset_dir))
                caption = {
                    "file_name": file_name,
                    "text": " ".join(f.readlines()).strip(),
                }

            captions.append(caption)

        else:
            print(f"no captions for {txt_file}")
    return captions


# Convert .txt captions to metadata.jsonl file for dataset
def setup_metadata(output, captions):
    captions = load_captions(output, captions)

    if len(captions) == 0:
        raise ValueError("yo no captions")

    # print(json.dumps(captions, indent=4))

    print("Saving captions")

    with open(Path(output) / "metadata.jsonl", "w") as f:
        # jsonl has json for each item
        for item in captions:
            f.write(json.dumps(item) + "\n")


def shuffle_caption(text, shuffle_on=", ", frozen_parts=1, dropout=0.0):
    parts = [part.strip() for part in text.split(shuffle_on)]

    # we want to keep the first part of the text, but shuffle the rest
    frozen = []

    if frozen_parts > 0:
        for i in range(frozen_parts):
            frozen.append(parts.pop(0))

    final_parts = []

    if dropout > 0.0:
        final_parts = []
        for part in parts:
            rand = random.random()
            if rand > dropout:
                final_parts.append(part)
    else:
        final_parts = parts

    random.shuffle(final_parts)

    return shuffle_on.join(frozen + final_parts)


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


def collate_fn(batch):
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


# def preprocess_train(example_batch):
#     """Apply train_transforms across a batch."""
#     example_batch["pixel_values"] = [
#         train_transforms(image.convert("RGB")) for image in example_batch["image"]
#     ]
#     return example_batch


seed = 42
torch.manual_seed(seed)
random.seed(seed)

processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    device_map="auto",
)


## SETUP DATASET

setup_metadata(Path(dataset_dir), captions=[])

ds = load_dataset(
    "imagefolder",
    num_proc=4,
    data_dir=dataset_dir,
    split="train",
    # split="validation",
    # split=["train", "test"],
)

ds = ds.train_test_split(test_size=0.2, shuffle=True)

batch_size = 8
shuffle_captions = True
frozen_parts = 1
caption_dropout = 0.1

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
    frozen_parts=frozen_parts,
    caption_dropout=caption_dropout,
    shuffle_captions=shuffle_captions,
)
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    num_workers=4,
    batch_size=batch_size,
    collate_fn=collate_fn,
)

val_dataset = ImageCaptioningDataset(ds["test"], processor)
val_dataloader = DataLoader(
    val_dataset,
    shuffle=False,
    num_workers=4,
    batch_size=batch_size,
    collate_fn=collate_fn,
)


print(ds["train"])

print(ds["train"][0]["text"])
print(ds["train"][0]["image"])

# next(iter(train_dataloader))
# next(iter(train_dataloader))
#
# raise ValueError("we gotta go")

## SETUP MODEL

# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", load_in_8bit=True
# )

# blip_config = BlipConfig(
#     {"attention_dropout": 0.1, "hidden_dropout_prob": 0.0}, {"attention_dropout": 0.1}
# )
blip_config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-large")
blip_config.hidden_dropout_prob = 0.2
blip_config.attention_probs_dropout_prob = 0.3

model = BlipForConditionalGeneration.from_pretrained(
    # "Salesforce/blip-image-captioning-base"
    "Salesforce/blip-image-captioning-large",
    device_map={"": 0},
    config=blip_config,
)

peft_module = "LoRA"

if peft_module == "LoRA":
    ## LORA

    print(model)
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

model.enable_input_require_grads()

if peft_module == "IA3":
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

## TRAINING

epochs = 5
save_every_n_epochs = 5
gradient_accumulation_steps = 2
gradient_checkpointing = True

## OPTIMIZERS

## PRODIGY

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

## DADAPTATION

# from dadaptation import DAdaptAdam

# lr = 1.0
# weight_decay = 0.05
# optimizer_args = {"lr": lr, "decouple": True}
# print("Using DAdaptation optimizer")
# print(optimizer_args)
# optimizer = DAdaptAdam(model.parameters(), weight_decay=weight_decay, **optimizer_args)

## ADAM
# lr = 2e-3
# weight_decay = 1e-4
# optimizer_args = {"lr": lr, "weight_decay": weight_decay}
# optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)


## SCHEDULER


# scheduler = torch.optim.lr_scheduler.CyclicLR(
#     optimizer, 2e-5, 2e-3, step_size_up=100, step_size_down=100, cycle_momentum=False
# )

# CosineAnnealingLR

n_epochs = 1
steps = n_epochs * (len(train_dataset) / (batch_size * gradient_accumulation_steps))
scheduler_args = {"steps": steps}
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

# CosineAnnealingWarmRestarts

n_epochs = 1
steps = int(
    n_epochs * (len(train_dataset) / (batch_size * gradient_accumulation_steps))
)
t_mult = 2
scheduler_args = {"steps": steps, "t_mult": t_mult}
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, steps, T_mult=t_mult
)
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

if gradient_checkpointing:
    model.gradient_checkpointing_enable()

accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    log_with="wandb",
)

project_config = {
    "lr": lr,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "weight_decay": weight_decay,
    "optimizer": type(optimizer).__name__,
    "optimizer_args": optimizer_args,
    "scheduler": type(scheduler).__name__,
    "scheduler_args": scheduler_args,
    "target_modules": target_modules,
    "gradient_checkpointing": gradient_checkpointing,
    "peft_config": type(peft_config).__name__,
}

if type(peft_config).__name__ == "LoraConfig":
    project_config = {
        **project_config,
        "lora_dropout": lora_dropout,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }

if type(peft_config).__name__ == "IA3Config":
    project_config = {**project_config, "feedforward_modules": ff_modules}

accelerator.init_trackers(project_name=training_name, config=project_config)

model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, scheduler
)

max_grad_norm = 1.0

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = accelerator.device
#
# print(f"Use {device}")
#
# model.to(device)

model.train()

scaler = torch.cuda.amp.GradScaler()

global_step = 0

loss_list = []
loss_total = 0.0

val_loss_list = []
val_loss_total = 0.0

for epoch in range(epochs):
    print("Epoch:", epoch + 1)
    # for idx, batch in enumerate(train_dataloader):
    t_dataloader = tqdm(train_dataloader)
    for step, batch in enumerate(t_dataloader):
        with accelerator.accumulate(model):
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

            # print("Loss:", loss.item())
            # print(f"dlr: {optimizer.param_groups[0].get('d')}")
            accelerator.backward(loss)
            # loss.backward()

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Unscales gradients and calls
            # or skips optimizer.step()
            # scaler.step(optimizer)
            optimizer.step()

            scheduler.step()
            # scaler.update()
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

    sample(model, processor)

    # Save every n epochs
    if save_every_n_epochs % (epoch + 1) == 0:
        accelerator.wait_for_everyone()
        save_epoch_to = f"{save_to}_{epoch+1}"
        print(f"Saved to {save_epoch_to}")
        model.save_pretrained(save_epoch_to, safe_serialization=True)


accelerator.end_training()

# Save only if we haven't saved before
if save_every_n_epochs % (epochs + 1) != 0:
    print(f"Saved to {save_to}")
    model.save_pretrained(save_to, safe_serialization=True)


