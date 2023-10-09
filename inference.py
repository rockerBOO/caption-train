import argparse
import glob

from PIL import Image
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BlipForConditionalGeneration,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # processor = AutoProcessor.from_pretrained("microsoft/git-base")
    # model = AutoModelForCausalLM.from_pretrained(args.model)

    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.float16,
    )

    model = BlipForConditionalGeneration.from_pretrained(
        # "Salesforce/blip-image-captioning-base"
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(model, args.lora_model)

    model.to(device)
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
    for idx, img in enumerate(images):
        image = Image.open(img)
        inputs = processor(images=image, return_tensors="pt").to(
            device, dtype=torch.float16
        )
        pixel_values = inputs.pixel_values

        generated_ids = model.generate(
            pixel_values=pixel_values, max_length=args.max_token_length
        )
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        results.append({"idx": idx, "img": img, "caption": generated_caption})
        print(idx, img, generated_caption)

    # Save a CSV of the results
    import csv

    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["idx", "img", "caption"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Save captions next to images with a .txt file
    for result in results:
        img = Path(result["img"])

        with open(img.with_name(img.stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(result["caption"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Disabling for now
    # parser.add_argument(
    #     "--base_model",
    #     type=str,
    #     default="microsoft/git-base",
    #     help="Model path",
    # )

    parser.add_argument(
        "--lora_model",
        type=str,
        help="LoRA model directory for the base model",
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Image files to use",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=75,
        help="Maximum number of tokens to generate",
    )

    args = parser.parse_args()
    main(args)
