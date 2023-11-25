import argparse
import random
from pathlib import Path
from time import gmtime, strftime
from typing import List
import math
import wandb

import tomllib
import torch
import torchvision.transforms as T
from accelerate import Accelerator
from datasets import load_dataset

# from evaluate import load
from peft import IA3Config, LoraConfig, get_peft_model
from prodigyopt import Prodigy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
)
from tqdm import tqdm
from transformers import (
    AutoConfig,
    # AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    BlipForConditionalGeneration,
    # GitModel,
    GitForCausalLM,
)
from torchmetrics.multimodal.clip_score import CLIPScore

from caption_train.captions import setup_metadata, shuffle_caption

# from websockets.sync.client import connect


@torch.no_grad()
def sample(example, model, processor, accelerator):
    # with accelerator.autocast():
    #     inputs = processor(images=example["image_filename"],
    #     return_tensors="pt").to(
    #         accelerator.device
    #     )
    # pixel_values = inputs.pixel_values

    with accelerator.autocast(), torch.inference_mode():
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
                processed_batch[key] = torch.stack(
                    [example[key] for example in batch]
                )
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch],
                    padding=True,
                    return_tensors="pt",
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs[
                    "attention_mask"
                ]
        return processed_batch

    return process_collate_fn


# def send_captions_images(images, captions):
#     req = dict(
#         req_from="trainer",
#         type="captions",
#         payload=dict(images=images, captions=captions),
#     )
#
#     broadcast(json.dumps(req))


def setup_dataset(processor, args):
    setup_metadata(Path(args.dataset_dir), captions=[])

    ds = load_dataset(
        "imagefolder",
        # num_proc=4,
        data_dir=args.dataset_dir,
        split="train",
    )

    print(f"Validation split {args.validation_split}")

    if args.validation_split > 0.0:
        if args.test_split > 0.0:
            # Split between train and test datasets
            ds = ds.train_test_split(
                test_size=args.validation_split + args.test_split,
                shuffle=True,
                seed=args.seed,
            )

            print(((args.validation_split / args.test_split) / 2))
            # Split between validation and test datasets
            test_dataset = ds["test"].train_test_split(
                test_size=((args.validation_split / args.test_split) / 2),
                shuffle=True,
                seed=args.seed,
            )

            validation_dataset = (
                test_dataset["train"] if args.validation_split > 0.0 else []
            )
            test_dataset = (
                test_dataset["test"] if args.test_split > 0.0 else []
            )
        else:
            # Split between train and validation datasets
            validation_dataset = ds.train_test_split(
                test_size=args.validation_split, shuffle=True, seed=args.seed
            )
            test_dataset = []

        print(validation_dataset)
        print(test_dataset)

    train_ds = ds["train"] if args.validation_split > 0.0 else ds

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
        train_ds,
        processor,
        train_transforms,
        frozen_parts=args.frozen_parts,
        caption_dropout=args.caption_dropout,
        shuffle_captions=args.shuffle_captions,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        # num_workers=4,
        batch_size=args.batch_size,
        collate_fn=collate_fn(processor),
    )

    val_dataset = ImageCaptioningDataset(validation_dataset, processor)

    val_dataloader = DataLoader(
        val_dataset if args.validation_split > 0.0 else [],
        shuffle=False,
        # num_workers=4,
        batch_size=args.batch_size,
        collate_fn=collate_fn(processor),
    )

    test_dataset = ImageCaptioningDataset(validation_dataset, processor)

    test_dataloader = DataLoader(
        test_dataset if args.test_split > 0.0 else [],
        shuffle=False,
        # num_workers=4,
        batch_size=args.batch_size,
        collate_fn=collate_fn(processor),
    )

    print(train_ds)

    print(train_ds[0]["text"])
    print(train_ds[0]["image"])

    return (
        train_dataset,
        train_dataloader,
        val_dataset,
        val_dataloader,
        test_dataset,
        test_dataloader,
    )


def wrap_in_ia3(model, args):
    print("Using IA3")
    # -- IA3

    # for key in model.state_dict().keys():
    #     print(key)

    # qkv, projection, fc1, fc2, query, key, value, dense, decoder
    # target_modules = ["query", "value"]
    target_modules = ".*encoder.*(self_attn|self|crossattention).*(qkv|key|value|dense|projection).*"
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


MODEL_TO_LORA_MODULES = {
    "BlipForConditionalGeneration": ".*encoder.*(self_attn|self|crossattention|attention).*(qkv|key|query|value|projection).*",
    "BlipModel": ".*encoder.*(self_attn|self|crossattention|attention).*(qkv|key|query|value|projection).*",
    "GitModel": ".*encoder.*(self_attn|self|crossattention|attention).*(k_proj|v_proj|q_proj|out_proj|query|key|value).*",
}

# (
#     ".*encoder.*(self_attn|self|crossattention|attention).*(qkv|key|query|value|projection).*"
#     # ".*encoder.*(self_attn|self|crossattention|attention).*(k_proj|v_proj|q_proj|out_proj|query|key|value).*"
#     # "query",
#     # "key",
#     # "qkv",
#     # "crossattention.output.dense",
#     # "attention.output.dense",
#     # "self_attn.projection",
# )


def wrap_in_lora(model, args):
    print("Using LoRA")
    ## LORA

    # lora_rank = 32
    # lora_alpha = 16
    # lora_dropout = 0.05
    # qkv, projection, fc1, fc2, query, key, value, dense, decoder
    target_modules = MODEL_TO_LORA_MODULES[type(model.base_model).__name__]

    peft_args = {
        **{"bias": "none", "target_modules": target_modules},
        **args.peft_args,
    }

    peft_config = LoraConfig(
        # r=lora_rank,
        # lora_alpha=lora_alpha,
        # lora_dropout=lora_dropout,
        # target_modules=target_modules,
        **peft_args,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, peft_config


def get_scheduler(optimizer, train_dataset_length, args):
    # Placeholder scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda v: v]
    )
    scheduler_args = args.scheduler_args

    if args.lr_scheduler == "CyclicLR":
        scheduler_args = {
            **{
                "t_max": 2e-5,
                "T_min": 2e-3,
                "step_size_up": 100,
                "step_size_down": 100,
                "cycle_momentum": False,
            },
            **scheduler_args,
        }
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, **scheduler_args
        )

    if args.lr_scheduler == "OneCycleLR":
        epochs = args.epochs

        scheduler_args = {
            **{
                "steps_per_epoch": int(
                    train_dataset_length
                    / (args.batch_size * args.gradient_accumulation_steps)
                ),
                "epochs": epochs,
            },
            **scheduler_args,
        }
        print(scheduler_args)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, **scheduler_args
        )

    # CosineAnnealingLR

    # we can calculate steps by the arguments
    if args.lr_scheduler == "CosineAnnealingLR":
        n_epochs = args.epochs
        print(n_epochs * train_dataset_length)
        steps = (n_epochs * train_dataset_length) / (
            args.batch_size * args.gradient_accumulation_steps
        )
        steps = n_epochs * train_dataset_length
        print(
            "CosineAnnealingLR",
            f"({n_epochs} * {train_dataset_length}) / ({args.batch_size} * {args.gradient_accumulation_steps})",
        )
        scheduler_args = {**{"T_max": steps}, **scheduler_args}
        print(scheduler_args)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **scheduler_args
        )
        print(scheduler)

    if args.lr_scheduler == "CosineAnnealingWarmRestarts":
        n_epochs = 1
        steps = int(
            n_epochs
            * (
                train_dataset_length
                / (args.batch_size * args.gradient_accumulation_steps)
            )
        )
        t_mult = 2
        scheduler_args = {
            **{"steps": steps, "T_mult": t_mult},
            **scheduler_args,
        }
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **scheduler_args
        )

    if scheduler is None:
        raise RuntimeError("Invalid scheduler")

    return scheduler, scheduler_args


def get_optimizer(model, args):
    # OPTIMIZERS

    optimizer = None
    optimizer_args = args.optimizer_args

    # PRODIGY

    if args.optimizer == "Prodigy":
        #
        optimizer_args = {
            **optimizer_args,
        }
        print("Using Prodigy optimizer")
        print(optimizer_args)
        optimizer = Prodigy(
            model.parameters(), lr=args.learning_rate, **optimizer_args
        )

    # DADAPTATION

    if args.optimizer == "DAdaptAdam":
        from dadaptation import DAdaptAdam

        lr = 1.0
        weight_decay = 0.05
        optimizer_args = {"lr": lr, "decouple": True}
        print("Using DAdaptation optimizer")
        print(optimizer_args)
        optimizer = DAdaptAdam(
            model.parameters(), weight_decay=weight_decay, **optimizer_args
        )

    # ADAM
    if args.optimizer == "AdamW":
        lr = args.learning_rate
        # weight_decay = 1e-4
        optimizer_args = {**{"lr": lr}, **optimizer_args}
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)

    if optimizer is None:
        raise RuntimeError("Invalid optimizer")

    return optimizer, optimizer_args


def compute_metrics(eval_pred, processor, wer):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(
        predicted, skip_special_tokens=True
    )
    wer_score = wer.compute(
        predictions=decoded_predictions, references=decoded_labels
    )
    return {"wer_score": wer_score}


def get_blip_model(args):
    blip_config = AutoConfig.from_pretrained(args.model_name_or_path)

    # TODO: map model_config to blip_config
    blip_config.hidden_dropout_prob = 0.2
    blip_config.attention_probs_dropout_prob = 0.3

    model = BlipForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        device_map={"": 0},
        config=blip_config,
    )

    return model


def get_git_model(args):
    git_config = AutoConfig.from_pretrained(args.model_name_or_path)

    git_config.hidden_dropout_prob = 0.2
    git_config.attention_probs_dropout_prob = 0.3

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": 0},
        config=git_config,
    )

    return model


def get_auto_model(args):
    auto_config = AutoConfig.from_pretrained(
        args.model_name_or_path, **args.model_config_args
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=auto_config,
        # device_map="auto",
        # torch_dtype=torch.float16,
    )

    return model


def process_batch(batch, model, processor, accelerator, args):
    # input_ids = batch.pop("input_ids").to(accelerator.device)
    # pixel_values = batch.pop("pixel_values").to(accelerator.device)
    # attention_mask = batch.pop("attention_mask").to(accelerator.device)

    # print(batch.keys())
    #
    # for k in batch.keys():
    #     print(k, batch[k].shape)

    with accelerator.autocast():
        # labels = input_ids
        # if isinstance(model.base_model, GitModel):
        #     kwargs = {"labels": labels}
        # else:
        #     kwargs = {}

        # outputs = model(
        #     input_ids=input_ids,
        #     pixel_values=pixel_values,
        #     attention_mask=attention_mask,
        #     **kwargs,
        # )
        outputs = model(**batch)

        logits = outputs["logits"]

        # print(outputs.keys())

        # print(processor.vision
        # vision_config = model.base_model.image_encoder.vision_config

        # print('base model', type(model.get_base_model()).__name__)
        # Blip
        if isinstance(model.get_base_model(), BlipForConditionalGeneration):
            vision_config = model.base_model.vision_model.config
            text_config = model.base_model.text_decoder.config
            vocab_size = text_config.vocab_size  # 30522

        # GIT
        if isinstance(model.get_base_model(), GitForCausalLM):
            vision_config = model.get_base_model().git.config.vision_config
            vocab_size = model.get_base_model().config.vocab_size  # 30522

        # print(vision_config)

        # patch_size = 16  # git base coco
        # patch_size = 14  # git large coco
        # patch_size = 16
        # git_image_size = 224
        # image_size = 384

        patch_size = vision_config.patch_size
        image_size = vision_config.image_size
        num_image_tokens = int((image_size / patch_size) ** 2 + 1)

        if isinstance(model.get_base_model(), GitForCausalLM):
            num_image_tokens = (
                model.get_base_model()
                .git.encoder.layer[0]
                .attention.self.image_patch_tokens
            )

        # print("num_image_tokens", num_image_tokens)
        # print("logits", logits.shape)

        if "input_ids" in batch:
            # we are doing next-token prediction; shift prediction scores and input ids by one

            if isinstance(model.get_base_model(), GitForCausalLM):
                shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()
            else:
                shifted_logits = logits[:, :-1, :].contiguous()

            # print("shifted_logits", shifted_logits.shape)
            labels = batch["input_ids"].clone()
            # print("labels", labels.shape)
            labels = labels[:, 1:].contiguous()
            # print("labels contiguous", labels.shape)
            # Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            # 'none': no reduction will be applied, 'mean': the weighted mean of the
            # output is taken, 'sum': the output will be summed.
            reduction = "mean"  # 'none', 'mean', 'sum'

            # Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing.
            # https://arxiv.org/abs/1512.00567
            label_smoothing = 0.1

            loss_fct = nn.CrossEntropyLoss(
                reduction=reduction, label_smoothing=label_smoothing
            )

            shifted_logits_view = shifted_logits.view(-1, vocab_size)
            labels_view = labels.view(-1)

            # torch.Size([0, 30522]) torch.Size([30])
            # print(shifted_logits_view.shape, labels_view.shape)

            loss = loss_fct(shifted_logits_view, labels_view)

            if reduction == "none":
                loss = loss.view(outputs["logits"].size(0), -1).sum(1)

        # feat = outputs["logits"]
        # target = input_ids.clone()
        # # need_predict = torch.tensor([[0] + [1] * len(target) + [1]])
        # # need_predict = torch.Tensor().to(accelerator.device)
        #
        # feat = feat[:, :-1].contiguous()
        # target = target[:, 1:].contiguous()
        # # print(target.shape)
        # # print(feat.shape)
        # # need_predict = need_predict[:, 1:].contiguous()
        #   # Git TransformerDecoderTextualHead text_decoder vocab size
        # feat = feat.view(-1, vocab_size)
        # target = target.view(-1)
        #
        # print(target.shape)
        # print(feat.shape)
        # # need_predict = need_predict.view(-1)
        # # print(need_predict.shape)
        # #
        # # valid_mask = need_predict == 1
        # # print(valid_mask.shape)
        # # target = target[valid_mask]
        # # feat = feat[valid_mask]
        #
        # loss = nn.CrossEntropyLoss(reduction="none")(feat, target)

        # log_probs = -nn.functional.log_softmax(logits, dim=-1)
        #
        # print(labels)
        #
        # print(log_probs)
        # if labels.dim() == log_probs.dim() - 1:
        #     labels = labels.unsqueeze(-1)
        #
        # nll_loss = nn.functional.nll_loss(log_probs, labels, reduction="none")
        #
        # print(nll_loss)

        # ignore_index: int = -100
        # padding_mask = labels.eq(ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        # labels = torch.clamp(labels, min=0)
        #
        # nll_loss = log_probs.gather(dim=-1, index=labels)
        # smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        #
        # nll_loss.masked_fill_(padding_mask, 0.0)
        # smoothed_loss.masked_fill_(padding_mask, 0.0)
        #
        # num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        # nll_loss = nll_loss.sum() / num_active_elements
        # smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        #
        # epsilon = 0.1
        # loss = (1 - epsilon) * nll_loss + epsilon * smoothed_loss

        # print(outputs.keys())
        #
        # import sys
        #
        # sys.exit(2)

        if args.interactive is True:
            # TODO: Not ready yet
            # captions = processor.batch_decode(input_ids, skip_special_tokens=True)
            # images = [Image.fromarray(pixels) for pixels in pixel_values]

            # send_captions_images(images, captions)
            #
            # send_and_wait_for_response()

            # update run with response data

            print("interactive")
    # print(outputs.keys())
    # print(nll_loss)
    # loss = loss.mean()

    # print(f"batch {batch.size()}")
    # for key, item in batch.items():
    #     print(f"{key} {item.size()}")
    #
    # print(
    #     f"loss {loss.size()} {loss.item()} {loss.item()/batch['pixel_values'].size(0)}"
    # )
    return loss.mean()


def step_log(logs, lr_scheduler, optimizer, accelerator, args):
    logs = {
        "lr/lr": lr_scheduler.get_last_lr()[0],
        # "lr/lr": args.learning_rate,
        **logs,
    }

    if "d" in optimizer.param_groups[0]:
        logs["lr/d*lr"] = (
            # optimizer.param_groups[0].get("d") * scheduler.get_last_lr()[0]
            optimizer.param_groups[0].get("d")
            * args.learning_rate
        )

    # Plot momentum
    if args.lr_scheduler == "CyclicLR" or args.lr_scheduler == "OneCycleLR":
        logs["momentum/betas1"] = optimizer.param_groups[0]["betas"][0]

    return logs


# def sample(example, model, processor, accelerator):
@torch.no_grad()
def evaluate(example, model, processor, metric, accelerator):
    # We will use CLIPScore with the captions from the image and
    # compare it with the CLIP score from the caption and the image

    # - lets make a caption

    # with accelerator.autocast():
    #     inputs = processor(images=example["image_filename"],
    #     return_tensors="pt").to(
    #         accelerator.device
    #     )
    # pixel_values = inputs.pixel_values

    with accelerator.autocast(), torch.inference_mode():
        generated_ids = model.generate(
            pixel_values=example["pixel_values"],
            num_beams=3,
            min_length=3,
            max_length=75,
        )
        generated_captions = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        score = metric(
            example["pixel_values"].clamp(0, 1),
            generated_captions,
        )
        score.detach().item()

        return score, generated_captions


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


def train(
    batch_size,
    model,
    processor,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    args,
):
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

    if peft_module == "LoRA":
        model, peft_config = wrap_in_lora(model, args)
    elif peft_module == "IA3":
        model, peft_config = wrap_in_ia3(model, args)
    else:
        raise ValueError(f"Invalid PEFT module: {peft_module}")

    # if args.interactive:
    #     websocket = connect("ws://localhost:8765")
    #     websocket.send(dict(req_from="trainer", type="connect"))

    # OPTIMIZER
    optimizer, optimizer_args = get_optimizer(model, args)

    # SCHEDULER
    lr_scheduler, lr_scheduler_args = get_scheduler(
        optimizer, len(train_dataloader.dataset), args
    )

    # OUTPUT
    output_dir = Path(args.output_dir)

    training_datetime = strftime("%Y-%m-%d-%H%M%S", args.training_start)
    save_to = output_dir / f"{args.training_name}-{training_datetime}"

    # ACCELERATOR
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.log_with,
    )

    def clean_dict(unclean_dict):
        cleaned_dict = {}

        for key in unclean_dict.keys():
            v = unclean_dict.get(key)

            if type(v) not in [bool, str, int, float, dict]:
                continue

            cleaned_dict[key] = v

        return cleaned_dict

    project_config = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optimizer": type(optimizer).__name__,
        "optimizer_args": optimizer_args,
        "lr_scheduler": type(lr_scheduler).__name__,
        "lr_scheduler_args": lr_scheduler_args,
        "peft": type(peft_config).__name__,
        "peft_config": {**clean_dict(peft_config.__dict__)},
    }

    accelerator.init_trackers(
        project_name=args.training_name, config=project_config
    )

    if args.log_with in ["wandb", "all"]:
        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
        wandb_tracker.define_metric("current_step", hidden=True)
        wandb_tracker.define_metric("epoch_step", hidden=True)
        wandb_tracker.define_metric("validation_step", hidden=True)
        wandb_tracker.define_metric("loss/current", step_metric="current_step")
        wandb_tracker.define_metric(
            "loss/validation_current", step_metric="validation_step"
        )
        wandb_tracker.define_metric(
            "loss/validation_average", step_metric="epoch_step"
        )
        wandb_tracker.define_metric(
            "loss/epoch_average", step_metric="epoch_step"
        )

    print(f"train dataloader {len(train_dataloader)}  pre-accelerator")
    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )
    print(f"train dataloader {len(train_dataloader)} post-accelerator")

    print(
        f"({len(train_dataloader)} * {args.epochs}) / ({args.gradient_accumulation_steps})"
    )

    print(lr_scheduler)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
    
    if wandb is not None:
        # Magic
        wandb_tracker.watch(model, log_freq=100)

    # TRAIN
    model.train()

    # Should we compile here? lets see
    # model = torch.compile(model, backend="inductor")
    # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    # model = model.to_bettertransformer()

    progress_bar = tqdm(
        range(
            int(
                math.ceil(
                    (len(train_dataloader) * args.epochs)
                    / (args.gradient_accumulation_steps)
                )
            )
        ),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )
    global_step = 0

    loss_recorder = LossRecorder()
    val_loss_recorder = LossRecorder()

    # EPOCHS
    for epoch in range(args.epochs):
        accelerator.print(f"\nepoch {epoch+1}/{args.epochs}")

        # BATCH
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = process_batch(
                    batch, model, processor, accelerator, args
                )

                accelerator.backward(loss)

                if (
                    accelerator.sync_gradients
                    and args.max_grad_norm is not None
                    and args.max_grad_norm != 0.0
                ):
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

                logs = {
                    "current_step": step + (len(train_dataloader) * epoch),
                    "loss/current": loss.detach().item(),
                }

                accelerator.log(logs, step=global_step)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    current_loss = loss.detach().item()
                    loss_recorder.add(
                        epoch=epoch,
                        step=int(step / args.gradient_accumulation_steps),
                        loss=current_loss,
                    )
                    logs = {
                        "loss/average": loss_recorder.moving_average,
                    }

                    logs = step_log(
                        logs, lr_scheduler, optimizer, accelerator, args
                    )

                    accelerator.log(logs, step=global_step)

                    postfix = {
                        "avg_loss": loss_recorder.moving_average,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }

                    if "d" in optimizer.param_groups[0]:
                        postfix["d*lr"] = (
                            optimizer.param_groups[0].get("d")
                            * lr_scheduler.get_last_lr()[0]
                        )

                    progress_bar.set_postfix(postfix)

        accelerator.wait_for_everyone()

        # VALIDATION
        # ----------

        val_progress_bar = tqdm(
            range(len(val_dataloader)),
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc="validation steps",
        )
        v_dataloader = val_dataloader
        with torch.no_grad():
            for val_step, batch in enumerate(v_dataloader):
                val_progress_bar.update(1)
                loss = process_batch(
                    batch, model, processor, accelerator, args
                )

                current_loss = loss.detach().item()

                val_loss_recorder.add(
                    epoch=epoch, step=val_step, loss=current_loss
                )

                accelerator.log(
                    {
                        "validation_step": val_step
                        + (len(v_dataloader) * epoch),
                        "loss/validation_current": current_loss,
                    },
                    step=global_step,
                )

        print(f"Validation avg loss {val_loss_recorder.moving_average}")

        loggable = {
            "epoch_step": epoch + 1,
            "loss/validation_average": val_loss_recorder.moving_average,
        }

        accelerator.log(loggable, step=global_step)

        # load image
        for i, val in enumerate(val_dataloader):
            if i >= args.validation_samples or 5:
                break
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

        loggable = {
            "epoch_step": epoch + 1,
            "loss/epoch_average": loss_recorder.moving_average,
        }

        accelerator.log(loggable, step=global_step)

    accelerator.free_memory()

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    metric.to(accelerator.device)

    scores = []
    for step, example in enumerate(test_dataloader):
        score, caption = evaluate(
            example, model, processor, metric, accelerator
        )
        print(f"score: {score} - {caption}")
        scores.append((score, example["pixel_values"], caption))

    print(
        f"average CLIP score {sum([score[0] for score in scores])/len(scores)}"
    )

    if args.log_with in ["wandb", "all"]:
        # data = {
        #     "clip_scores": [score[0] for score in scores],
        #     "captions": [caption[1] for caption in scores],
        # }
        #
        # tbl = wandb.Table(columns=["image", "label"])
        #
        # images = np.random.randint(0, 255, [2, 100, 100, 3], dtype=np.uint8)
        # labels = ["panda", "gibbon"]
        # [tbl.add_data(wandb.Image(image), label) for image, label in zip(images, labels)]
        tbl = wandb.Table(columns=["clip_score", "image", "caption"])

        [
            tbl.add_data(score, wandb.Image(image, caption=caption), caption)
            for (score, image, caption) in scores
        ]
        accelerator.log({"clip_scores": tbl}, step=global_step)
        accelerator.log(
            {
                "average_clip_score": sum([score[0] for score in scores])
                / len(scores)
            },
            step=global_step,
        )

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

    args.training_start = gmtime()

    if "blip" in args.model_name_or_path:
        model = get_blip_model(args)
    elif "git" in args.model_name_or_path:
        model = get_git_model(args)
    # model = get_auto_model(args)
    # model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
    )

    # DATASET
    (
        train_dataset,
        train_dataloader,
        val_dataset,
        val_dataloader,
        test_dataset,
        test_dataloader,
    ) = setup_dataset(processor, args)

    print(f"Training: {len(train_dataloader)}")
    print(f"Validation: {len(val_dataloader)}")
    print(f"Test: {len(test_dataloader)}")

    train(
        args.batch_size,
        model,
        processor,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        args,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    # For example: training/sets/
    argparser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to output the resulting model to",
    )

    argparser.add_argument(
        "--dataset_dir",
        required=True,
        help="Directory for where the image/captions are stored. Is recursive.",
    )

    argparser.add_argument(
        "--learning_rate", type=float, help="Learning rate for the training"
    )

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
        "--shuffle_captions",
        action="store_true",
        help="Shuffle captions when training",
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

    argparser.add_argument(
        "--optimizer",
        choices=["Prodigy", "AdamW"],
        help="Optimizer to use",
    )

    argparser.add_argument(
        "--lr_scheduler",
        choices=[
            "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts",
            "CyclicLR",
            "OneCycleLR",
        ],
        help="Learning rate scheduler",
    )

    argparser.add_argument(
        "--scheduler_args",
        default={},
        help="Arguments for the learning rate scheduler",
    )

    argparser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for the validation dataset",
    )

    argparser.add_argument(
        "--validation_samples",
        type=int,
        default=5,
        help="Number of samples to make of the validation dataset",
    )

    argparser.add_argument(
        "--log_with", type=str, default=None, help="Log with"
    )
    argparser.add_argument("--peft_args", default={}, help="PEFT args")

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

    argparser.add_argument(
        "--config_file",
        default=None,
        help="Config file with all the arguments",
    )

    args = argparser.parse_args()

    if args.config_file is not None:
        with open(args.config_file, "rb") as f:
            toml_args = tomllib.load(f)

            tmp = {
                **vars(args),
                **toml_args,
            }
            args = argparse.Namespace(**tmp)
            print(args)

    main(args)
