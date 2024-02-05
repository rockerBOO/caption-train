# Caption Trainer

Train captioning models (image to text) using hugging face compatible models (models that use text+image pairs and produces text).

## Support

- BLIP

## Install

Suggested: Create a `venv` for this project, and activate the `venv`.

```
python -m venv .venv
source .venv/bin/activate # .venv/bin/activate.bat on windows
```

Then install the dependencies. You will also need [PyTorch](https://pytorch.org/get-started/locally/) for your hardware.

`pip install -r requirements.txt`

## Usage

`python train2.py --help`

```bash
$ python train2.py --help
usage: train2.py [-h] --output_dir OUTPUT_DIR --dataset_dir DATASET_DIR [--lr LR] [--model_name_or_path MODEL_NAME_OR_PATH]
                 [--training_name TRAINING_NAME] --epochs EPOCHS [--save_every_n_epochs SAVE_EVERY_N_EPOCHS] [--batch_size BATCH_SIZE]
                 [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--gradient_checkpointing] [--max_grad_norm MAX_GRAD_NORM]
                 [--shuffle_captions] [--frozen_parts FROZEN_PARTS] [--caption_dropout CAPTION_DROPOUT] [--peft_module {LoRA,IA3}]
                 [--seed SEED] [--interactive INTERACTIVE] [--model_config MODEL_CONFIG] [--optimizer {Prodigy,AdamW}]
                 [--lr_scheduler {CosineAnnealingLR,CosineAnnealingWarmRestarts,CyclicLR,OneCycleLR}] [--scheduler_args SCHEDULER_ARGS]
                 [--config_file CONFIG_FILE]

options:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Directory to output the resulting model to
  --dataset_dir DATASET_DIR
                        Directory for where the image/captions are stored. Is recursive.
  --lr LR               Learning rate for the training
  --model_name_or_path MODEL_NAME_OR_PATH
                        Model name on hugging face or path to model. (Should be a BLIP model at the moment)
  --training_name TRAINING_NAME
                        Training name used in creating the output directory and in logging (wandb)
  --epochs EPOCHS       Number of epochs to run the training for
  --save_every_n_epochs SAVE_EVERY_N_EPOCHS
                        Save an output of the current epoch every n epochs
  --batch_size BATCH_SIZE
                        Number of image/text pairs to train in the same batch
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Gradient accumulation steps
  --gradient_checkpointing
                        Use gradient checkpointing
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm
  --shuffle_captions    Shuffle captions when training
  --frozen_parts FROZEN_PARTS
                        How many parts (parts separated by ',') do we want to keep in place when shuffling
  --caption_dropout CAPTION_DROPOUT
                        Amount of parts we dropout in the caption.
  --peft_module {LoRA,IA3}
                        PEFT module to use in training
  --seed SEED           Seed for the random number generation
  --interactive INTERACTIVE
                        Interactive hook for modifying the dataset while running
  --model_config MODEL_CONFIG
                        Model configuration parameters (dropout, device_map, ...)
  --optimizer {Prodigy,AdamW}
                        Optimizer to use
  --lr_scheduler {CosineAnnealingLR,CosineAnnealingWarmRestarts,CyclicLR,OneCycleLR}
                        Learning rate scheduler
  --scheduler_args SCHEDULER_ARGS
                        Arguments for the learning rate scheduler
  --config_file CONFIG_FILE
                        Config file with all the arguments
```

Example `config.toml`. Very raw usage of parameters and can be any value also in help.

```toml
train_dir = "/mnt/900/input/tmp/30_woodcut_style/"
seed = 1337

[optimizer_args]
weight_decay = 0.1

[scheduler_args]
base_lr = 1.0
max_lr = 1.1
```

Logs to wandb.

---

## Test

```bash
$ pytest
============================= test session starts ==============================
platform linux -- Python 3.11.5, pytest-7.4.3, pluggy-1.3.0
rootdir: /home/rockerboo/code/caption-train
collected 2 items

iterative/test_schema.py ..                                              [100%]

============================== 2 passed in 0.31s ===============================
```

## Ideas

Implement [WiSE-ft](https://github.com/mlfoundations/wise-ft) merging for keeping robustness after fine-tuning
