## Run BLIP training

```
$ uv run accelerate launch train_blip_peft.py --help
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
$ uv run python inference.py --help
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

uv run accelerate launch inference.py \
	--base_model=$base_model \
	--images $input \
	--beams=10 \
	--peft_model $lora_model \
    # --save_captions
```
