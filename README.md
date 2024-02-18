# Caption Trainer

Train captioning models (image to text) using hugging face compatible models (models that use text+image pairs and produces text).

## Support

- BLIP

## Install

Suggested: Create a `venv` for this project, and activate the `venv`.

### venv

```bash
python -m venv venv
source venv/bin/activate # venv\Scripts\activate.bat on windows
```

### Poetry

Using poetry for dependencies. Make sure it is installed.

```bash
pipx install poetry
```

Additional [Poetry install instructions](https://python-poetry.org/docs/#installation) if necessary.

### Dependencies

```bash
poetry install
```

Then install the dependencies. You will also need [PyTorch](https://pytorch.org/get-started/locally/) for your hardware.

## Usage

### Setup dataset from text/image pairs

```bash
$ poetry run python compile_captions.py --help
usage: compile_captions.py [-h] dataset_dir output_dir

        Create hugging face dataset compatible caption file

        Take Kohya-ss or image/text file pairs and compile it
        into a compatible file

        $ poetry run python compile_captions.py /path/to/captions/dir

        # Output metadata.jsonl to a different location
        $ poetry run python compile_captions.py /path/to/captions/dir /path/to/output_dir


positional arguments:
  dataset_dir
  output_dir

options:
  -h, --help   show this help message and exit
```

```bash
$ poetry run python compile_captions.py /path/to/captions/dir /path/to/output_dir
```

### Run training

```bash
$ poetry run python train4.py --help
usage: train4.py [-h] [--device DEVICE] [--model_id MODEL_ID] [--rank RANK] [--alpha ALPHA] [--dropout DROPOUT] [--target_modules TARGET_MODULES [TARGET_MODULES ...]] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                 [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                 dataset_dir output_dir

    Caption trainer for BLIP

    Designed to be used with Hugging Face datasets.

    ---

    Use compile_captions.py to create a compatible dataset from
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
