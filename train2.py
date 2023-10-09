from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    BlipForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
import torch
from pathlib import Path
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator
from time import gmtime, strftime

dataset_dir = "/mnt/900/input/cyberpunk-anime/processed/cyberpunk-edgerunners-768-cap/"
# dataset_dir = (
#     "/mnt/900/input/others/knface_v9/base_knface_lr_balanced_renamed_prefixed/"
# )

datetime = strftime("%Y-%m-%d-%H%M%S", gmtime())
save_to = f"training/sets/cyberpunk-edgerunners-{datetime}"

# check if saveto directory is there


def sample(model, processor):
    with torch.no_grad():
        # load image
        for i in range(10):
            example = ds["test"][i]
            image = example["image"]

            # prepare image for the model
            inputs = processor(images=image, return_tensors="pt").to(
                device, dtype=torch.float16
            )
            pixel_values = inputs.pixel_values

            generated_ids = model.generate(pixel_values=pixel_values, max_length=75)
            generated_caption = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            print(f"Train: {example['text']}")
            print(f"Generated caption: {generated_caption}")


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
            captions += load_captions(file, captions)
            continue

        if file.suffix not in [".png", ".jpg"]:
            continue

        # need to check for images and then get the associated .txt file

        txt_file = file.with_name(f"{file.stem}.txt")

        if txt_file.exists():
            with open(txt_file) as f:
                caption = {
                    "file_name": str(file.relative_to(dataset_dir)),
                    "text": " ".join(f.readlines()).strip(),
                }

            captions.append(caption)

    return captions


# Convert .txt captions to metadata.jsonl file for dataset
def setup_metadata(output, captions):
    captions = load_captions(output, captions)

    if len(captions) == 0:
        raise ValueError("yo no captions")

    # print(json.dumps(captions, indent=4))

    with open(Path(output) / "metadata.jsonl", "w") as f:
        # jsonl has json for each item
        for item in captions:
            f.write(json.dumps(item) + "\n")


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(
            images=item["image"],
            padding="max_length",
            return_tensors="pt",
        )
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
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


torch.manual_seed(42)

processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    torch_dtype=torch.float16,
)


## SETUP DATASET

setup_metadata(Path(dataset_dir), captions=[])

ds = load_dataset(
    "imagefolder",
    # num_proc=1,
    data_dir=dataset_dir,
    split="train",
    # split="validation",
    # split=["train", "test"],
)

ds = ds.train_test_split(test_size=0.1, shuffle=True)

batch_size = 1

train_dataset = ImageCaptioningDataset(ds["train"], processor)
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=collate_fn,
)

val_dataset = ImageCaptioningDataset(ds["test"], processor)
val_dataloader = DataLoader(
    val_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=collate_fn,
)


print(ds["train"])

print(ds["train"][0]["text"])
print(ds["train"][0]["image"])

## SETUP MODEL

# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", load_in_8bit=True
# )

model = BlipForConditionalGeneration.from_pretrained(
    # "Salesforce/blip-image-captioning-base"
    "Salesforce/blip-image-captioning-large",
    torch_dtype=torch.float16,
)

print(model)
lora_rank = 16
lora_alpha = 16
lora_dropout = 0.01
# lora_dropout = 0.01
target_modules = ["query", "key"]

# Let's define the LoraConfig
config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    # bias="none",
    # target_modules=["q_proj", "k_proj"],
    target_modules=target_modules,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

## TRAINING

epochs = 10
gradient_accumulation_steps = 1

## OPTIMIZERS

## PRODIGY

from prodigyopt import Prodigy

lr = 1.0
weight_decay = 0.01
optimizer_args = {
    "lr": lr,
    "weight_decay": weight_decay,
    "safeguard_warmup": True,
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
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


## SCHEDULER

# scheduler = torch.optim.lr_scheduler.CyclicLR(
#     optimizer, 2e-5, 2e-3, step_size_up=100, step_size_down=100, cycle_momentum=False
# )
# steps = 10
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=2e-5)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=int(len(train_dataset)/epochs), eta_min=0.5
# )

scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=1.2,
#     steps_per_epoch=int(len(train_dataloader) / gradient_accumulation_steps),
#     epochs=epochs,
# )

accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps, log_with="wandb"
)
accelerator.init_trackers(
    project_name="BLIP-FT-LoRA",
    config={
        "lora_dropout": lora_dropout,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "optimizer": type(optimizer).__name__,
        "target_modules": target_modules,
    },
)

model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, scheduler
)

max_grad_norm = 1.0

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = accelerator.device

print(f"Use {device}")

model.to(device)

model.train()


for epoch in range(epochs):
    print("Epoch:", epoch)
    # for idx, batch in enumerate(train_dataloader):
    t_dataloader = tqdm(train_dataloader)
    for batch in t_dataloader:
        with accelerator.accumulate(model):
            input_ids = batch.pop("input_ids").to(device)
            # attention_mask = batch.pop("attention_mask").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            # input_ids = batch.pop("input_ids")
            attention_mask = batch.pop("attention_mask").to(device)
            # pixel_values = batch.pop("pixel_values")

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss

            # print("Loss:", loss.item())
            # print(f"dlr: {optimizer.param_groups[0].get('d')}")
            accelerator.backward(loss)
            # loss.backward()

            # if accelerator.sync_gradients:
            #     accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # Log to wandb by calling `accelerator.log`, `step` is optional
            # loggable = {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
            loggable = {
                "loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "d*lr": optimizer.param_groups[0].get("d") * scheduler.get_last_lr()[0],
            }

            accelerator.log(loggable)
            # dataloader.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            t_dataloader.set_postfix(
                {
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "d*lr": optimizer.param_groups[0].get("d")
                    * scheduler.get_last_lr()[0],
                }
            )

    # avg_val_loss = 0

    v_dataloader = tqdm(val_dataloader)
    for batch in val_dataloader:
        with torch.no_grad():
            input_ids = batch.pop("input_ids").to(device)
            # attention_mask = batch.pop("attention_mask").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            # input_ids = batch.pop("input_ids")
            attention_mask = batch.pop("attention_mask").to(device)
            # pixel_values = batch.pop("pixel_values")

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss

            loggable = {"val_loss": loss.item()}

            accelerator.log(loggable)

            v_dataloader.set_postfix({"val_loss": loss.item()})

    sample(model, processor)
    # plot()

accelerator.end_training()
model.save_pretrained(save_to)
