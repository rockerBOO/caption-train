import argparse
from accelerate.utils import set_seed
import torch
from peft import LoraConfig
from peft.mapping import get_peft_model

# from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)

from caption_train.util import get_group_args
from caption_train.opt import (
    get_accelerator,
    get_optimizer,
    get_scheduler,
    opt_config_args,
)
from caption_train.datasets import set_up_datasets, set_up_image_text_pair, datasets_config_args
from caption_train.trainer import (
    FileConfig,
    OptimizerConfig,
    PeftConfig,
    Trainer,
    TrainingConfig,
    training_config_args,
    peft_config_args,
)


def set_up_model(model_id, training_config: TrainingConfig, peft_config: PeftConfig):
    quantization_config = None
    if args.quantize:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quantization_config,
        revision="refs/pr/1",
        torch_dtype=torch.bfloat16,
    )

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
        print("Using activation checkpointing")
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
        print("Using gradient checkpointing")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    config = LoraConfig(
        r=peft_config.rank,
        lora_alpha=peft_config.alpha,
        lora_dropout=training_config.dropout,
        bias="none",
        # target specific modules (look for Linear in the model)
        # print(model) to see the architecture of the model
        target_modules=peft_config.target_modules,
        task_type="CAUSAL_LM",
        inference_mode=False,
        use_rslora=peft_config.rslora,
        init_lora_weights=peft_config.init_lora_weights,
    )

    # We layer our PEFT on top of our model using the PEFT config
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    trainable_modules = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_modules += 1

    print(f"Trainable modules: {trainable_modules}")

    return model, processor


# All the linear modules
FLORENCE_TARGET_MODULES = [
    "qkv",
    "proj",
    "k_proj",
    "v_proj",
    "q_proj",
    "out_proj",
    "fc1",
    "fc2",
    # "dw" conv
]


def main(
    args,
    training_config: TrainingConfig,
    peft_config: PeftConfig,
    optimizer_config: OptimizerConfig,
    dataset_config: FileConfig,
):
    if args.seed:
        set_seed(args.seed)

    print("Loading model and tokenizer")
    # Model
    model, processor = set_up_model(args.model_id, training_config, peft_config)

    # Accelerator
    accelerator = get_accelerator(args)
    model, processor = accelerator.prepare(model, processor)

    accelerator.print("Loading optimizer")
    # Optimizer
    optimizer = get_optimizer(model, training_config.learning_rate, optimizer_config)
    optimizer = accelerator.prepare(optimizer)

    # Scheduler
    scheduler = None

    accelerator.print("Loading dataset")
    dataset_loc = args.dataset or args.dataset_dir
    if dataset_loc.is_dir():
        datasets = set_up_image_text_pair(model, processor, accelerator, training_config, dataset_config)
    else:
        datasets = set_up_datasets(dataset_loc, processor, training_config, dataset_config)

    if args.scheduler:
        scheduler = get_scheduler(
            optimizer,
            training_config,
            args,
            steps_per_epoch=len(datasets.train_dataloader),
        )
        scheduler = accelerator.prepare(scheduler)

    datasets.accelerate(accelerator)

    accelerator.print("Start training")
    trainer = Trainer(
        model=model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        datasets=datasets,
        config=training_config,
        file_config=dataset_config,
    )
    trainer.train()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="""
        Florence 2 trainer


    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    argparser, peft_group = peft_config_args(argparser, FLORENCE_TARGET_MODULES)
    argparser, training_group = training_config_args(argparser)
    argparser, opt_group = opt_config_args(argparser)
    argparser, dataset_group = datasets_config_args(argparser)

    args = argparser.parse_args()


    training_config = TrainingConfig(**get_group_args(args, training_group))
    optimizer_config = OptimizerConfig(**get_group_args(args, opt_group))
    peft_config = PeftConfig(**get_group_args(args, peft_group))
    dataset_config = FileConfig(**get_group_args(args, dataset_group))

    main(args, training_config, peft_config, optimizer_config, dataset_config)
