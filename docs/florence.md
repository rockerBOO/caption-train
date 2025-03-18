# Florence 2

<!--toc:start-->

- [Florence 2](#florence-2)
  - [Florence 2 training](#florence-2-training)
  - [Florence 2 inference](#florence-2-inference)
  - [Florence 2 inference example](#florence-2-inference-example)
  - [Example Florence 2 bash script](#example-florence-2-bash-script)

<!--toc:end-->

Training of LoRA for Florence 2 base and large.

````
$ uv run python train_florence_peft.py --help
usage: train_florence_peft.py [-h] [--target_modules TARGET_MODULES [TARGET_MODULES ...]] [--rslora] [--rank RANK] [--alpha ALPHA] [--dropout DROPOUT] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                              [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--epochs EPOCHS] [--gradient_checkpointing] [--quantize] [--sample_every_n_epochs SAMPLE_EVERY_N_EPOCHS] [--sample_every_n_steps SAMPLE_EVERY_N_STEPS]
                              [--save_every_n_epochs SAVE_EVERY_N_EPOCHS] [--save_every_n_steps SAVE_EVERY_N_STEPS] [--device DEVICE] [--model_id MODEL_ID] [--seed SEED] [--log_with {all,wandb,tensorboard}] [--name NAME] [--shuffle_captions]
                              [--frozen_parts FROZEN_PARTS] [--caption_dropout CAPTION_DROPOUT] [--max_length MAX_LENGTH] [--prompt PROMPT] [--optimizer_name OPTIMIZER_NAME] [--scheduler {OneCycle}] [--accumulation_rank ACCUMULATION_RANK] [-ac]
                              [--accumulation-rank ACCUMULATION_RANK] [--optimizer_rank OPTIMIZER_RANK] [--optimizer_args OPTIMIZER_ARGS] [--dataset_dir DATASET_DIR] [--dataset DATASET] --output_dir OUTPUT_DIR

        Florence 2 trainer



options:
  -h, --help            show this help message and exit

PEFT:
  --target_modules TARGET_MODULES [TARGET_MODULES ...]
                        Target modules to be trained. Consider Linear and Conv2D modules in the base model. Default: qkv proj k_proj v_proj q_proj out_proj fc1 fc2.
  --rslora              RS LoRA scales alpha to size of rank
  --rank RANK           Rank/dim for the LoRA. Default: 4
  --alpha ALPHA         Alpha for scaling the LoRA weights. Default: 4

Training:
  --dropout DROPOUT     Dropout for the LoRA network. Default: 0.25
  --learning_rate LEARNING_RATE
                        Learning rate for the LoRA. Default: 1e-4
  --batch_size BATCH_SIZE
                        Batch size for the image/caption pairs. Default: 1
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --epochs EPOCHS       Number of epochs to run. Default: 5
  --gradient_checkpointing
                        Gradient checkpointing to reduce memory usage in exchange for slower training
  --quantize            Quantize the training model to 4-bit
  --sample_every_n_epochs SAMPLE_EVERY_N_EPOCHS
                        Sample the dataset every n epochs
  --sample_every_n_steps SAMPLE_EVERY_N_STEPS
                        Sample the dataset every n steps
  --save_every_n_epochs SAVE_EVERY_N_EPOCHS
                        Save the model every n epochs
  --save_every_n_steps SAVE_EVERY_N_STEPS
                        Save the model every n steps
  --device DEVICE       Device to run the training on. Default: cuda or cpu
  --model_id MODEL_ID   Model to train on. microsoft/Florence-2-base-ft or microsoft/Florence-2-large-ft. Default: microsoft/Florence-2-base-ft
  --seed SEED           Seed used for random numbers
  --log_with {all,wandb,tensorboard}
                        Log with. all, wandb, tensorboard
  --name NAME           Name to be used with saving and logging
  --shuffle_captions    Shuffle captions when training
  --frozen_parts FROZEN_PARTS
                        How many parts (parts separated by ',') do we want to keep in place when shuffling
  --caption_dropout CAPTION_DROPOUT
                        Amount of parts we dropout in the caption.
  --max_length MAX_LENGTH
                        Max length of the input text. Defaults to max length of the Florence 2 processor.
  --prompt PROMPT       Task for florence

Optimizer:
  --optimizer_name OPTIMIZER_NAME
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
  --optimizer_args OPTIMIZER_ARGS
                        Optimizer args

Datasets:
  --dataset_dir DATASET_DIR
                        Dataset directory image and caption file pairs. img.jpg and img.txt
  --dataset DATASET     Dataset with the metadata.jsonl file and images
  --output_dir OUTPUT_DIR
                        Save the LoRA files to this directory```

## Florence 2 training

Train a Florence 2 LoRA on your dataset.

```bash
uv run accelerate launch train_florence_peft.py --dataset_dir /path/to/image-text-pairs --output_dir loras/my-lora-name
````

## Florence 2 inference

Inference on your Florence 2 LoRA model.

```
$ uv run python inference_florence.py --help
usage: inference_florence.py [-h] [--base_model BASE_MODEL] [--peft_model PEFT_MODEL] --images IMAGES [--seed SEED] [--task {<CAPTION>,<DETAILED_CAPTION>,<MORE_DETAILED_CAPTION>}] [--batch_size BATCH_SIZE] [--save_captions] [--overwrite] [--append]
                             [--caption_extension CAPTION_EXTENSION] [--max_token_length MAX_TOKEN_LENGTH] [--revision REVISION]

        Image captioning using Florence 2

        $ python inference.py --images /path/to/images/ --peft_model models/my_lora

        See --save_captions if producing an image/text file pair dataset.


options:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
                        Model to load from hugging face 'microsoft/Florence-2-base-ft'
  --peft_model PEFT_MODEL
                        PEFT (LoRA, IA3, ...) model directory for the base model. For transfomers models (this tool).
  --images IMAGES       Directory of images or an image file to caption
  --seed SEED           Seed for rng
  --task {<CAPTION>,<DETAILED_CAPTION>,<MORE_DETAILED_CAPTION>}
                        Task to run the captioning on.
  --batch_size BATCH_SIZE
                        Number of images to caption in the batch
  --save_captions       Save captions to the images next to the image.
  --overwrite           Overwrite caption files with new caption. Default: We skip writing captions that already exist
  --append              Append captions to the captions that already exist or write the caption.
  --caption_extension CAPTION_EXTENSION
                        Extension to save the captions as, .txt, .caption
  --max_token_length MAX_TOKEN_LENGTH
                        Maximum number of tokens to generate
  --revision REVISION   Revision of the model to load. Default: None
```

## Florence 2 inference example

Inference example for Florence 2. Works with PEFT generated models.

```bash
$ uv run accelerate launch inference.py --images /path/to/images/ --peft_model models/my_peft
```

## Example Florence 2 bash script

```bash
input="/path/to/images/to/caption"
peft_model="training/review/2024-02-13-232352-15f12fb9-e20-l-2.87"

echo "Input: $input"
echo "PEFT Model: $peft_model"

uv run accelerate launch inference_florence.py \
	--images $input \
	--peft_model $peft_model \
    # --save_captions
```
