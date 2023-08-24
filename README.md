# Caption Trainer

Train captioning models (image to text) using hugging face compatible models (models that use text+image pairs and produces text).

## Support

- GIT

## Install

Suggested: To create a `venv` for this project, and activate the `venv`.

Then install the dependencies

`pip install -r requirements.txt`

## Usage

`python train.py --help`

```
$ python train.py --help
usage: train.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH] [--debug] [--epochs EPOCHS] [--seed SEED]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--batch-size BATCH_SIZE] [--save_every_n_epochs SAVE_EVERY_N_EPOCHS] [--lr LR]
                [--base_lr BASE_LR] [--max_lr MAX_LR] [--network_rank NETWORK_RANK] [--network_alpha NETWORK_ALPHA] [--network_dropout NETWORK_DROPOUT]
                [--logging_steps LOGGING_STEPS] [--config CONFIG] [--lora_bits LORA_BITS] [--block_size BLOCK_SIZE]

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Model name from hugging face or path to model
  --debug               Debug the captions
  --epochs EPOCHS       Epochs to run
  --seed SEED           Seed to run the training with
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Gradient accumulation steps
  --batch-size BATCH_SIZE
                        Batch size
  --save_every_n_epochs SAVE_EVERY_N_EPOCHS
                        save every n epochs. default to save only the last epoch
  --lr LR               Learning rate (not currently used)
  --base_lr BASE_LR     Learning rate optimizer minumum learning rate for CyclicLR
  --max_lr MAX_LR       Learning rate optimizer max learning rate for CyclicLR
  --network_rank NETWORK_RANK
                        Network rank for LoRA
  --network_alpha NETWORK_ALPHA
                        Network alpha for LoRA
  --network_dropout NETWORK_DROPOUT
                        Network dropout for LoRA
  --logging_steps LOGGING_STEPS
                        When using logging, how often to log
  --config CONFIG       Config file in toml
  --lora_bits LORA_BITS
                        Bits for LoRA 4, 8 bit quantization. Defaults to 16 bit
  --block_size BLOCK_SIZE
                        Block size, defaults to 2048
```

Example `config.toml`. Very raw usage of parameters and can be any value also in help.

```toml
train_dir="/mnt/900/input/tmp/30_woodcut_style/"
seed = 1337

[optimizer_args]
weight_decay = 0.1

[scheduler_args]
base_lr = 1.0
max_lr = 1.1
```

Logs to wandb.

---

## Ideas

Implement [WiSE-ft](https://github.com/mlfoundations/wise-ft) merging for keeping robustness after fine-tuning
