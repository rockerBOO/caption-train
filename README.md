# Caption Trainer

<!--toc:start-->
- [Caption Trainer](#caption-trainer)
  - [Support](#support)
  - [Install](#install)
    - [venv](#venv)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Setup dataset from text/image pairs](#setup-dataset-from-textimage-pairs)
  - [Run BLIP training](#run-blip-training)
  - [BLIP LoRA inference](#blip-lora-inference)
  - [Example BLIP LoRA usage bash script](#example-blip-lora-usage-bash-script)
  - [Florence 2](#florence-2)
  - [Florence 2 training](#florence-2-training)
  - [Florence 2 inference](#florence-2-inference)
  - [Florence 2 inference example](#florence-2-inference-example)
  - [Example Florence 2 bash script](#example-florence-2-bash-script)
  - [Development](#development)
    - [Test](#test)
<!--toc:end-->

Train IQA for captioning using 🤗 compatible models (models that use text+image pairs and produces text).

## Support

- BLIP
- [Florence 2](#florence-2)
- WIP Moondream 2

## Install

Suggested: Create a `venv` for this project, and activate the `venv`.

### venv

```bash
python -m venv venv
source venv/bin/activate # venv\Scripts\activate.bat on windows
```

### Dependencies

With the `venv` activated.

```
pip install -r requirements.txt
```

Or with Poetry.

```bash
poetry install
```

Additional [Poetry install instructions](https://python-poetry.org/docs/#installation).

_NOTE_ You will also need [PyTorch](https://pytorch.org/get-started/locally/) for your hardware.

## Usage

We need to make sure your images have a compatible dataset. We have a script to convert over image/text captioned file pairs.

Convert:

- images/img1.jpg
- images/img1.txt

To a `metadata.jsonl` using `compile_captions.py` which is compatible with 🤗Datasets.

- Find the appropriate model you want to train a LoRA on, BLIP, Florence 2, Moondream 2.
- Then we can train on that dataset.
- Also have inference scripts for each model.

### Setup dataset from text/image pairs

```
$ python compile_captions.py --help
usage: compile_captions.py [-h] dataset_dir output_dir

        Create hugging face dataset compatible caption file

        Take Kohya-ss or image/text file pairs and compile it
        into a compatible file

        $ python compile_captions.py /path/to/captions/dir

        # Output metadata.jsonl to a different location
        $ python compile_captions.py /path/to/captions/dir /path/to/output_dir


positional arguments:
  dataset_dir
  output_dir

options:
  -h, --help   show this help message and exit
```

```bash
$ python compile_captions.py /path/to/captions/dir /path/to/output_dir
```

## Run BLIP training

```
$ accelerate launch train_blip_network.py --help
usage: train_blip_network.py [-h] [--device DEVICE] [--model_id MODEL_ID] [--rank RANK] [--alpha ALPHA] [--dropout DROPOUT] [--target_modules TARGET_MODULES [TARGET_MODULES ...]] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                 [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                 dataset_dir output_dir

    Caption trainer for BLIP

    Designed to be used with Hugging Face datasets.

    ---

    Use `compile_captions.py` to create a compatible dataset from
    image/text pairing.

    Example: a.png a.txt

    Creates a datasets compatible metadata.jsonl from those pairings.


positional arguments:
  dataset_dir           Dataset directory with the metadata.jsonl file and images
  output_dir            Save the LoRA files to this directory

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to run the training on. Default: cuda or cpu
  --model_id MODEL_ID   Model to train on. Salesforce/blip-image-captioning-base or Salesforce/blip-image-captioning-large. Default: Salesforce/blip-image-captioning-base
  --rank RANK           Rank/dim for the LoRA. Default: 16
  --alpha ALPHA         Alpha for scaling the LoRA weights. Default: 32
  --dropout DROPOUT     Dropout for the LoRA network. Default: 0.05
  --target_modules TARGET_MODULES [TARGET_MODULES ...]
                        Target modules to be trained. Consider Linear and Conv2D modules in the base model. Default: self.query self.key self.value output.dense self_attn.qkv self_attn.projection mlp.fc1 mlp.fc2
  --learning_rate LEARNING_RATE
                        Learning rate for the LoRA. Default: 1e-3
  --weight_decay WEIGHT_DECAY
                        Weight decay for the AdamW optimizer. Default: 1e-4
  --batch_size BATCH_SIZE
                        Batch size for the image/caption pairs. Default: 2
  --epochs EPOCHS       Number of epochs to run. Default: 5
```

## BLIP LoRA inference

```
$ python inference.py --help
usage: inference.py [-h] [--base_model BASE_MODEL] [--peft_model PEFT_MODEL] --images IMAGES [--beams BEAMS] [--seed SEED] [--save_captions] [--caption_extension CAPTION_EXTENSION] [--max_token_length MAX_TOKEN_LENGTH]

options:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
                        Model to load from hugging face 'Salesforce/blip-image-captioning-base'
  --peft_model PEFT_MODEL
                        PEFT (LoRA, IA3, ...) model directory for the base model
  --images IMAGES       Directory of images or image file to caption
  --beams BEAMS         Save captions to the images next to the image
  --seed SEED           Seed
  --save_captions       Save caption files (.txt or caption extension) next to the image
  --caption_extension CAPTION_EXTENSION
                        Extension to save the captions as
  --max_token_length MAX_TOKEN_LENGTH
                        Maximum number of tokens to generate
```

## Example BLIP LoRA usage bash script

```bash
input="/path/to/images/to/caption"
lora_model="training/review/2024-02-13-232352-15f12fb9-e20-l-2.87"
base_model="Salesforce/blip-image-captioning-large"

echo "Input: $input"
echo "LoRA Model: $lora_model"
echo "Base Model: $base_model"

accelerate launch inference.py \
	--base_model=$base_model \
	--images $input \
	--beams=10 \
	--peft_model $lora_model \
    # --save_captions
```

## Florence 2

Training of LoRA for Florence 2 base and large.

```
$ python train_florence_lora.py --help
usage: train_florence_lora.py [-h] [--dataset_dir DATASET_DIR] [--output_dir OUTPUT_DIR] [--device DEVICE] [--model_id MODEL_ID] [--seed SEED] [--log_with {all,wandb,tensorboard}] [--name NAME] [--rank RANK] [--alpha ALPHA]
                              [--dropout DROPOUT] [--target_modules TARGET_MODULES [TARGET_MODULES ...]] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--batch_size BATCH_SIZE]
                              [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--epochs EPOCHS] [--gradient_checkpointing] [--quantize] [--accumulation_compression] [--rslora]
                              [--sample_every_n_epochs SAMPLE_EVERY_N_EPOCHS] [--sample_every_n_steps SAMPLE_EVERY_N_STEPS] [--save_every_n_epochs SAVE_EVERY_N_EPOCHS] [--save_every_n_steps SAVE_EVERY_N_STEPS]
                              [--optimizer_name {AdamW,AdamW8bit,Flora}] [--scheduler {OneCycle}] [--accumulation_rank ACCUMULATION_RANK] [-ac] [--accumulation-rank ACCUMULATION_RANK] [--optimizer_rank OPTIMIZER_RANK]

        Florence 2 trainer



options:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR
                        Dataset directory with the metadata.jsonl file and images
  --output_dir OUTPUT_DIR
                        Save the LoRA files to this directory
  --device DEVICE       Device to run the training on. Default: cuda or cpu
  --model_id MODEL_ID   Model to train on. microsoft/Florence-2-base-ft or microsoft/Florence-2-large-ft. Default: microsoft/Florence-2-base-ft
  --seed SEED           Seed used for random numbers
  --log_with {all,wandb,tensorboard}
                        Log with. all, wandb, tensorboard
  --name NAME           Name to be used with saving and logging
  --rank RANK           Rank/dim for the LoRA. Default: 16
  --alpha ALPHA         Alpha for scaling the LoRA weights. Default: 32
  --dropout DROPOUT     Dropout for the LoRA network. Default: 0.05
  --target_modules TARGET_MODULES [TARGET_MODULES ...]
                        Target modules to be trained. Consider Linear and Conv2D modules in the base model. Default: qkv proj k_proj v_proj q_proj out_proj.
  --learning_rate LEARNING_RATE
                        Learning rate for the LoRA. Default: 1e-4
  --weight_decay WEIGHT_DECAY
                        Weight decay for the AdamW optimizer. Default: 1e-4
  --batch_size BATCH_SIZE
                        Batch size for the image/caption pairs. Default: 1
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --epochs EPOCHS       Number of epochs to run. Default: 5
  --gradient_checkpointing
                        Gradient checkpointing to reduce memory usage in exchange for slower training
  --quantize            Quantize the training model to 4-bit
  --accumulation_compression
                        Accumulation compression for FloraAccelerator
  --rslora              RS LoRA scales alpha to size of rank
  --sample_every_n_epochs SAMPLE_EVERY_N_EPOCHS
                        Sample the dataset every n epochs
  --sample_every_n_steps SAMPLE_EVERY_N_STEPS
                        Sample the dataset every n steps
  --save_every_n_epochs SAVE_EVERY_N_EPOCHS
                        Save the model every n epochs
  --save_every_n_steps SAVE_EVERY_N_STEPS
                        Save the model every n steps
  --optimizer_name {AdamW,AdamW8bit,Flora}
                        Optimizer to use
  --scheduler {OneCycle}
                        Scheduler to use
  --accumulation_rank ACCUMULATION_RANK
                        Rank to use with FloraAccelerator for low rank optimizer
  -ac, --activation_checkpointing
                        Activation checkpointing using the FloraAccelerator
  --accumulation-rank ACCUMULATION_RANK
                        Accumulation rank for low rank accumulation
  --optimizer_rank OPTIMIZER_RANK
                        Flora optimizer rank for low-rank optimizer
```

## Florence 2 training

```bash
accelerate launch train_florence_lora.py --dataset_dir /path/to/image-text-pairs --output_dir loras/my-lora-name
```

## Florence 2 inference

```
$ python inference_florence.py --help
usage: inference_florence.py [-h]
                             [--base_model {microsoft/Florence-2-base-ft,microsoft/Florence-2-large-ft,microsoft/Florence-2-large,microsoft/Florence-2-base}]
                             [--peft_model PEFT_MODEL] --images IMAGES [--seed SEED] [--batch_size BATCH_SIZE] [--save_captions]
                             [--overwrite] [--caption_extension CAPTION_EXTENSION] [--max_token_length MAX_TOKEN_LENGTH]

        Train a LoRA on a Florence model for image captioning.

        $ python inference.py --images /path/to/images/ --peft_model models/my_lora


options:
  -h, --help            show this help message and exit
  --base_model {microsoft/Florence-2-base-ft,microsoft/Florence-2-large-ft,microsoft/Florence-2-large,microsoft/Florence-2-base}
                        Model to load from hugging face 'microsoft/Florence-2-base-ft'
  --peft_model PEFT_MODEL
                        PEFT (LoRA, IA3, ...) model directory for the base model. For transfomers models (this tool).
  --images IMAGES       Directory of images or image file to caption
  --seed SEED           Seed for rng
  --batch_size BATCH_SIZE
                        Number of images to caption
  --save_captions       Save captions to the images next to the image.
  --overwrite           Overwrite caption files
  --caption_extension CAPTION_EXTENSION
                        Extension to save the captions as
  --max_token_length MAX_TOKEN_LENGTH
                        Maximum number of tokens to generate
```

## Florence 2 inference example

Inference example

```bash
$ accelerate launch inference.py --images /path/to/images/ --peft_model models/my_lora
```

## Example Florence 2 bash script

```bash
input="/path/to/images/to/caption"
lora_model="training/review/2024-02-13-232352-15f12fb9-e20-l-2.87"
base_model="microsoft/Florence-2-base-ft"

echo "Input: $input"
echo "LoRA Model: $lora_model"

python inference_florence.py \
	--base_model=$base_model \
	--images $input \
	--peft_model $lora_model \
    # --save_captions
```

## Development

Poetry for dependencies.
Ruff for linting and formatting.
Pytest for testing.

### Test

```
$ pytest
============================= test session starts ==============================
platform linux -- Python 3.11.5, pytest-7.4.3, pluggy-1.3.0
rootdir: /home/rockerboo/code/caption-train
collected 2 items

iterative/test_schema.py ..                                              [100%]

============================== 2 passed in 0.31s ===============================
```
