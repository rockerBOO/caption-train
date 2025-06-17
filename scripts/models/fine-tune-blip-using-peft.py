# Install dependencies
#
# python -m venv venv
# source ./venv/bin/activate # linux
# call ./venv/scripts/Activate.bat  # windows?
#
# pip install transformers peft datasets
#
# Use the PyTorch instructions for your machine:
# [Get started — PyTorch](https://pytorch.org/get-started/locally/)
#
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model

from datasets import load_dataset

from torch.utils.data import DataLoader, Dataset

import torch.optim as optim

model_id = "Salesforce/blip-image-captioning-base"

# Set the device which works for you or use these defaults
device = "cuda" if torch.cuda.is_available() else "cpu"

# loading the model in float16
model = AutoModelForVision2Seq.from_pretrained(model_id)
model.to(device)

# Print the model to see the model architecture. PEFT works with Linear and Conv2D layers
# print(model)

processor = AutoProcessor.from_pretrained(model_id)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # target specific modules (look for Linear in the model)
    # print(model) to see the architecture of the model
    target_modules=[
        "self.query",
        "self.key",
        "self.value",
        "output.dense",
        "self_attn.qkv",
        "self_attn.projection",
        "mlp.fc1",
        "mlp.fc2",
    ],
)

# We layer our PEFT on top of our model using the PEFT config
model = get_peft_model(model, config)
model.print_trainable_parameters()


# We are extracting the train dataset
dataset = load_dataset("ybelkada/football-dataset", split="train")

print(len(dataset))
print(next(iter(dataset)))


# Each dataset may have different names and format and this function should be
# adjusted for each dataset. Creating `image` and `text` keys to correlate with
# the collator
class FootballImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding


def collator(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
            processed_batch[key].to(device)
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch],
                padding=True,
                return_tensors="pt",
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


train_dataset = FootballImageCaptioningDataset(dataset, processor)

# Set batch_size to the batch that works for you.
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)

# Setup AdamW
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)


# Set the model as ready for training, makes sure the gradient are on
model.train()


for epoch in range(50):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        loss = outputs.loss

        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if idx % 10 == 0:
            generated_output = model.generate(pixel_values=pixel_values, max_new_tokens=64)
            print(processor.batch_decode(generated_output, skip_special_tokens=True))


print("Saving to ./training/caption")

# hugging face models saved to the directory
model.save_pretrained("./training/caption")
