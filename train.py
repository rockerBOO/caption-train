import os
import sys
import glob
import json

# from tomlparse import argparse
import argparse
from pathlib import Path
from argparse import Namespace

import toml
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    GitProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from prodigyopt import Prodigy

captions = []


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset: DatasetDict, processor: GitProcessor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(
            images=item["image"],
            text=item["text"],
            padding="max_length",
            # padding=True,
            return_tensors="pt",
        )

        if args.augment_images:
            trans = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomCrop(180),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(
                        180, scale=(0.8, 1.2), ratio=(1.0, 1.0)
                    ),
                ]
            )
        else:
            trans = lambda _: _

        # remove batch dimension
        encoding = {k: trans(v.squeeze()) for k, v in encoding.items()}

        return encoding


def load_captions(images: list[str]) -> list[str]:
    # Load up captions
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    for image in images:
        # find the captoin
        file = Path(image)
        caption_file = Path(str(file.parent.absolute()) + "/" + file.stem + ".txt")
        if caption_file.exists():
            with open(caption_file) as f:
                captions.append(
                    {
                        # file_name removing images/ from path
                        "file_name": file.name,
                        # read all the lines of the file
                        "text": " ".join(f.readlines()).strip(),
                    }
                )
        else:
            print(f"No caption for {file}")

    print(json.dumps(captions, indent=4))
    return captions


def load_from_toml(toml_file):
    with open(toml_file, "r") as f:
        results = toml.load(f)

        print(results)
        return results

    return {}


def main(args):
    print(args)
    checkpoint = args.model_name_or_path or "microsoft/git-base"
    root = Path(args.train_dir)
    images = sum(
        [glob.glob(str(root.absolute()) + f"/*.{f}") for f in ["jpg", "jpeg", "png"]],
        [],
    )

    set_seed(args.seed or 1337)

    print(f"model: {args.model_name_or_path}")

    captions = load_captions(images)

    if len(captions) == 0:
        raise ValueError("yo no captions")

    with open(str(root.absolute()) + "/metadata.jsonl", "w") as f:
        for item in captions:
            f.write(json.dumps(item) + "\n")

    processor: GitProcessor = AutoProcessor.from_pretrained(checkpoint)

    dataset = load_dataset(
        "imagefolder",
        num_proc=1,
        data_dir=root.absolute(),
    )

    dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True)

    print(dataset)

    print(f"Train images: {len(dataset['train'])}")
    print(f"Test images: {len(dataset['test'])}")

    def transforms(example_batch):
        images = [x for x in example_batch["image"]]
        captions = [x for x in example_batch["text"]]
        inputs = processor(images=images, text=captions, padding="max_length")
        inputs.update({"labels": inputs["input_ids"]})
        return inputs

    dataset["train"].set_transform(transforms)
    dataset["test"].set_transform(transforms)
    train_dataset = dataset["train"]

    ## Tokenizer
    ## ------

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path or checkpoint)

    # add pad tokens and resize max length of the tokenizer because the model is trained using GPT2 tokenizer but has a longer max length
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # logger.info("Setting `pad_token` to `eos_token`: %s", tokenizer.eos_token)
        print(f"Setting `pad_token` to `eos_token`: {tokenizer.eos_token}")

    if args.block_size is None:
        # logger.info("Setting `block_size` 2048 since it was not set")
        print("Setting `block_size` 2048 since it was not set")
        tokenizer.model_max_length = 2048
    else:
        # logger.info("Setting `block_size` to %d", data_args.block_size)
        print(f"Setting `block_size` to {args.block_size}")
        tokenizer.model_max_length = args.block_size

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)

    ## Lora Quantization
    ## -----------------

    if "lora_bits" in args:
        if args.lora_bits == 4:
            print("Using QLoRA")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif args.lora_bits == 8:
            print("Using 8bit LoRA")

            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            print("Using 16bit LoRA")
            bnb_config = None
    else:
        print("Using 16bit LoRA")
        bnb_config = None

    ## Config
    ## ------

    config = AutoConfig.from_pretrained(checkpoint)

    # Things that were changed from the huggingface file
    config.gradient_checkpointing = False  # no gradient checkpointing for lora
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, quantization_config=bnb_config, config=config
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "fc1",
            "fc2",
            "query",
            "key",
            "value",
            "dense",
        ],
        inference_mode=False,
        r=args.network_rank,
        lora_alpha=args.network_alpha,
        lora_dropout=args.network_dropout,
    )

    model = get_peft_model(model, peft_config, adapter_name="CausalLM-LoRA")
    model.print_trainable_parameters()

    learning_rate = args.lr

    finished_steps = 0
    total_steps = args.epochs * len(train_dataloader)
    progress_bar = tqdm(range(total_steps), smoothing=0, desc="steps")

    optimizer_args = {"lr": learning_rate, **args.optimizer_args}

    # optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    print("Using Prodigy optimizer")
    print(optimizer_args)
    optimizer = Prodigy(model.parameters(), **optimizer_args)

    print("Using CyclicLR scheduler")
    print(args.scheduler_args)
    gas = args.gradient_accumulation_steps or 0
    bs = args.batch_size
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        #                              BS  GA  range
        step_size_up=int(total_steps / bs / gas / 2),
        step_size_down=int(total_steps / bs / gas / 2),
        **args.scheduler_args,
    )

    model_name = checkpoint.split("/")[1]

    training_args = TrainingArguments(
        output_dir=f"outputs/{model_name}-povblowjobpose",
        overwrite_output_dir=True,
        # learning_rate=5e-5,
        # learning_rate=3e-3,
        learning_rate=args.lr,
        # lr_scheduler_type=SchedulerType.COSINE,
        num_train_epochs=args.epochs,
        fp16=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=3,
        evaluation_strategy="epoch",
        # eval_steps=50,
        save_strategy="epoch",
        # save_every_n_epochs=1,
        # save_steps=50,
        logging_steps=args.logging_steps,
        seed=args.seed,
        save_safetensors=True,
        remove_unused_columns=False,
        # push_to_hub=True,
        label_names=["labels"],
        # load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        optimizers=(optimizer, scheduler),
    )

    trainer.train()


def train_model(args, model, optimizer, scheduler, train_dataloader, progress_bar):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda"
    model.to(device)

    model.train()

    step = 0

    for epoch in range(args.epochs):
        total_loss = 0
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids")
            pixel_values = batch.pop("pixel_values")
            labels = batch.pop("labels")

            # labels = input_ids
            #
            kwargs = {
                **batch,
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "labels": labels,
            }

            outputs = model(**kwargs)

            # if args.sample_steps and finished_steps % args.sample_steps == 0:
            #     with torch.no_grad():
            #         generated_ids = model.generate(
            #             pixel_values=pixel_values, max_length=75
            #         )
            #
            #         generated_captions = processor.batch_decode(
            #             generated_ids, skip_special_tokens=True
            #         )
            #         print(generated_captions)

            # # load image
            # example =
            # image = example["image"]
            # caption = sample_caption(image, processor, model, device=device)
            # print(caption)
            #
            # caption = processor.batch_decode(input_ids, skip_special_tokens=True)[0]
            # print(f"Caption: {caption}")

            # generated_ids = model.generate(
            #     pixel_values=pixel_values, max_length=75
            # )
            #
            # generated_captions = processor.batch_decode(
            #     generated_ids, skip_special_tokens=True
            # )
            # print(generated_captions)

            # print(outputs.keys())
            # loss = sum(outputs['logits'])
            loss = outputs.loss

            # print("loss")
            # print(f"loss: {loss}")

            total_loss += loss.detach().float()

            # print("Loss:", loss.item())

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # correct = 0
            # total = 0
            # assert len(eval_preds) == len(
            #     dataset["train"]['text']
            # ), f"{len(eval_preds)} != {len(dataset['train']['text'])}"
            # for pred, true in zip(eval_preds, dataset["train"]['text']):
            #     if pred.strip() == true.strip():
            #         correct += 1
            #     total += 1
            # accuracy = correct / total * 100

            writer.add_scalar("loss", loss, step)
            for i, lr in enumerate(scheduler.get_last_lr()):
                writer.add_scalar(f"lr-{i}", lr, step)

            progress_bar.update(1)
            progress_bar.set_postfix(**{"loss": float(loss)})
            finished_steps = finished_steps + 1

        if (
            epoch != 0
            and args.save_every_n_epochs
            and epoch % args.save_every_n_epochs == 0
        ):
            out = f"outputs/model-{epoch}"
            print(f"saving to {out}")
            model.save_pretrained(out)

        # Final epoch
        if epoch + 1 == args.epochs:
            out = "outputs/model"
            print(f"saving final to {out}")
            model.save_pretrained(out)

        step = step + 1

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        train_epoch_loss = train_epoch_loss.detach().float()
        writer.add_scalar(f"{train_ppl=} {train_epoch_loss=}", step)

        progress_bar.set_postfix(
            **{
                "loss": float(loss),
                "train_ppl": train_ppl,
                "train_epoch_loss": train_epoch_loss,
            }
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="microsoft/git-base",
        help="Model name from hugging face or path to model",
    )

    # parser.add_argument(
    #     "--train_dir", required=True, help="Directory with training data"
    # )

    parser.add_argument("--debug", action="store_true", help="Debug the captions")

    parser.add_argument("--epochs", type=int, default=50, help="Epochs to run")

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to run the training with"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )

    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        help="save every n epochs. default to save only the last epoch",
    )

    # parser.add_argument(
    #     "--max_token_length",
    #     default=75,
    #     type=int,
    #     help="max token length for the caption",
    # )
    # parser.add_argument(
    #     "--augment_images", action="store_true", help="Augment images for training"
    # )

    parser.add_argument(
        "--lr", type=float, default=1, help="Learning rate (not currently used)"
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=1.0,
        help="Learning rate optimizer minumum learning rate for CyclicLR",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=1.1,
        help="Learning rate optimizer max learning rate for CyclicLR",
    )
    # parser.add_argument("--dropout", type=float, default=0.0, help="Learning rate")
    parser.add_argument(
        "--network_rank", type=int, default=8, help="Network rank for LoRA"
    )
    parser.add_argument(
        "--network_alpha", type=int, default=4, help="Network alpha for LoRA"
    )
    parser.add_argument(
        "--network_dropout", type=float, default=0.1, help="Network dropout for LoRA"
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="When using logging, how often to log",
    )
    #
    parser.add_argument("--config", type=str, help="Config file in toml")

    parser.add_argument(
        "--lora_bits",
        type=int,
        help="Bits for LoRA 4, 8 bit quantization. Defaults to 16 bit",
    )

    parser.add_argument("--block_size", help="Block size, defaults to 2048")

    args = parser.parse_args()

    if args.config is not None:
        kwargs = {**vars(args), **load_from_toml(args.config)}
        args = Namespace(**kwargs)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
