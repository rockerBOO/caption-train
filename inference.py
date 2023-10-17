import argparse
import glob

from PIL import Image
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipImageProcessor
)
from peft import PeftModel
from pathlib import Path
import csv
from accelerate import Accelerator
from datasets import load_dataset
import torch
# from itertools import batched


@torch.no_grad()
def main(args):
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    # processor = BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

    model = BlipForConditionalGeneration.from_pretrained(args.base_model)

    accelerator = Accelerator()
    device = accelerator.device

    model = PeftModel.from_pretrained(model, args.peft_model)

    model, processor = accelerator.prepare(model, processor)

    model.eval()

    images = sum(
        [
            glob.glob(str(Path(args.images_dir).absolute()) + f"/*.{f}")
            for f in ["jpg", "jpeg", "png", "webp", "avif", "bmp"]
        ],
        [],
    )

    print(f"{len(images)} images")

    results = []
    images.sort()
    # for idx, img in enumerate(batched(images, 2)):
    batch_size = 16
    for i in range(0, len(images), batch_size):
        batch = [Image.open(img) for img in images[i : i + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(accelerator.device)
        with accelerator.autocast():
            generated_ids = model.generate(
                pixel_values=inputs.pixel_values, min_length=3, max_length=args.max_token_length
            )

            generated_captions = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        for idx, img in enumerate(images[i : i + batch_size]):
            results.append({"img": img, "caption": generated_captions[idx]})
            print(Path(img).stem, generated_captions[idx])

    if args.save_captions:
        # Save a CSV of the results
        with open("results.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["idx", "img", "caption"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        # Save captions next to images with a .txt file
        for result in results:
            img = Path(result["img"])

            with open(
                img.with_name(img.stem + args.caption_extension), "w", encoding="utf-8"
            ) as f:
                f.write(result["caption"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        help="Model to load from hugging face 'Salesforce/blip-image-captioning-large'",
    )

    parser.add_argument(
        "--peft_model",
        type=str,
        help="PEFT (LoRA, IA3, ...) model directory for the base model",
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory of images to caption",
    )

    parser.add_argument(
        "--save_captions",
        action="store_true",
        help="Save captions to the images next to the image",
    )

    parser.add_argument(
        "--caption_extension", default=".txt", help="Extension to save the captions as"
    )

    parser.add_argument(
        "--max_token_length",
        type=int,
        default=75,
        help="Maximum number of tokens to generate",
    )

    args = parser.parse_args()
    main(args)
