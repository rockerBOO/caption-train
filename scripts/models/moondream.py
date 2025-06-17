import argparse
from pathlib import Path
import math

import torch

# from peft import prepare_model_for_kbit_training
# from transformers import BitsAndBytesConfig (unused)
from einops import rearrange
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from torch.utils.data import DataLoader
from bitsandbytes.optim import Adam8bit

# from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729


class CaptionDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"],  # Should be a PIL image
            "qa": [
                {
                    "question": "Describe this image using simplified but vivid language. How would this image be described as an image caption?",
                    "answer": sample["caption"],
                }
            ],
        }


def lr_schedule(lr, step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * lr + 0.9 * lr * x / 0.1
    else:
        return 0.1 * lr + 0.9 * lr * (1 + math.cos(math.pi * (x - 0.1))) / 2


def main(args):
    # model_id = "vikhyatk/moondream2"
    model_id = "/home/rockerboo/code/others/moondream2"
    revision = "2024-04-02"

    # Quantization configuration (currently unused)
    # quantized_config = BitsAndBytesConfig(...)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # local_files_only=True,
        # trust_remote_code=True,
        revision=revision,
        # device_map="auto",
        device_map={"": 0},
        torch_dtype=torch.float16,
        # load_in_8bit=True,
        # quantization_config=nf4_config,
        # low_cpu_mem_usage=True,
        # load_in_4bit=True,
    )
    print(model.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    print(model)
    # print_gpu_utilization()

    # target_modules = ["qkv","fc1", "fc2", "proj", "Wqkv", "out_proj", "linear"]
    target_modules = ["mixer.Wqkv"]
    # target_modules = ["qkv"]
    # target_modules="all-linear",
    # target_modules = ["linear"]

    # model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=1,
        lora_alpha=1,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # target specific modules (look for Linear in the model)
        # print(model) to see the architecture of the model
        target_modules=target_modules,
    )

    config.inference_mode = False

    model.enable_input_require_grads()
    # model.text_model.transformer.embd.wte.requires_grad_(True)
    # model.text_model.transformer.gradient_checkpointing_enable()

    model = get_peft_model(model, config)
    # model.train()

    train(model, tokenizer, args)


def train(model, tokenizer, args):
    image_paths = []

    epochs = args.epochs or 5
    dtype = torch.float16
    lr = 3e-5
    grad_accum_steps = 1
    batch_size = 1

    use_wandb = False

    for image in args.images:
        path = Path(image)
        if path.is_dir():
            image_paths.extend([image for image in path.iterdir() if image.suffix != ".txt"])
        else:
            image_paths.append(image)

    captions = []
    for image in image_paths:
        with open(image.with_suffix(".txt"), "r") as f:
            captions.append(f.read())

    images = [Image.open(image) for image in image_paths]

    # questions = ["Describe this image with simplified language."] * len(images)

    dataset = []
    for caption, image in zip(captions, images):
        dataset.append({"caption": caption, "image": image})

    dataset = HFDataset.from_list(dataset)
    dataset = dataset.shuffle(seed=42)

    splits = dataset.train_test_split(test_size=0.2)

    subsets = splits["test"].train_test_split(test_size=0.5)
    subsets["val"] = subsets["train"]
    subsets["train"] = splits["train"]

    datasets = {
        "train": CaptionDataset(subsets["train"]),
        "val": CaptionDataset(subsets["train"]),
        "test": CaptionDataset(subsets["test"]),
    }

    def collate_fn(batch):
        images = [sample["image"] for sample in batch]
        images = torch.stack(model.vision_encoder.preprocess(images))
        images = rearrange(images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)

            for qa in sample["qa"]:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False,
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = tokenizer(f" {qa['answer']}{ANSWER_EOS}", add_special_tokens=False).input_ids
                toks.extend(a_t)
                labs.extend(a_t)

            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images.to(dtype=dtype),
            torch.stack([torch.tensor(token, dtype=torch.long) for token in tokens_acc]),
            torch.stack([torch.tensor(label, dtype=torch.long) for label in labels_acc]),
            torch.stack([torch.tensor(attn_mask, dtype=torch.bool) for attn_mask in attn_mask_acc]),
        )

    def compute_loss(batch, accelerator):
        images, tokens, labels, attn_mask = batch

        images = images.to(accelerator.device)
        tokens = tokens.to(accelerator.device)
        labels = labels.to(accelerator.device)
        attn_mask = attn_mask.to(accelerator.device)

        with torch.no_grad():
            img_embs = model.vision_encoder.encoder(images)
            img_embs = model.vision_encoder.projection(img_embs)

        tok_embs = model.text_model.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

        outputs = model.text_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attn_mask,
        )

        return outputs.loss

    # answers = model.batch_answer(images, prompts, tokenizer)

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            collate_fn=collate_fn,
        ),
    }

    model.text_model.train()

    total_steps = epochs * len(dataloaders["train"]) // grad_accum_steps
    optimizer = Adam8bit(
        [
            {"params": model.text_model.parameters()},
        ],
        lr=lr * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    from accelerate import Accelerator

    accelerator = Accelerator()

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, dataloaders["train"], dataloaders["val"]
    )

    i = 0
    for epoch in range(epochs):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            i += 1

            with accelerator.autocast():
                loss = compute_loss(batch, accelerator)

            # loss._requires_grad = True
            # loss._requires_grad = True
            # loss.backward()
            accelerator.backward(loss)

            if i % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(lr, i / grad_accum_steps, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            if i % 100 == 0 and use_wandb:
                # Calculate validation loss
                val_loss = 0
                for val_batch in tqdm(val_dataloader, desc="Validation"):
                    with torch.no_grad():
                        val_loss += compute_loss(val_batch).item()
                val_loss /= len(val_dataloader)

            if use_wandb:
                wandb.log(
                    {
                        "loss/train": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    | ({"loss/val": val_loss} if i % 100 == 0 else {})
                )

    model.save_pretrained("checkpoints/moondream-lora")

    model.eval()

    correct = 0

    for i, sample in enumerate(datasets["test"]):
        md_answer = model.answer_question(
            model.encode_image(sample["image"]),
            sample["qa"][0]["question"],
            tokenizer=tokenizer,
        )

        if md_answer == sample["qa"][0]["answer"]:
            correct += 1

        if i < 3:
            # display(sample['image'])
            print("Question:", sample["qa"][0]["question"])
            print("Ground Truth:", sample["qa"][0]["answer"])
            print("Moondream:", md_answer)

    print(f"\n\nAccuracy: {correct / len(datasets['test']) * 100:.2f}%")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""
    Generate captions using moondream VQA model.

    Uses fp16 model on CUDA.

    $ python moondream.py /path/to/image.png

    $ python moondream.py /path/to/image.png /path/to/image2.png

    $ python moondream.py /path/to/dir/images
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    argparser.add_argument(
        "images",
        nargs="+",
        help="List of image or image directories",
    )

    argparser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    argparser.add_argument("--epochs", type=int, default=5)

    args = argparser.parse_args()
    main(args)
