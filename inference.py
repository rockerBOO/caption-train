import argparse
import glob

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained(args.model)


    model = PeftModel.from_pretrained(
        model, "/home/rockerboo/code/caption-train/outputs/git-base-povblowjobpose/checkpoint-1850/CausalLM-LoRA/"
    )

    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     target_modules=["k_proj", "v_proj", "q_proj", "query", "key", "value"],
    #     inference_mode=False,
    #     r=args.network_rank,
    #     lora_alpha=args.network_alpha,
    #     lora_dropout=args.network_dropout,
    # )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

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
        inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        width, height = image.size
        image.resize((int(0.3 * width), int(0.3 * height)))

        # Runs the forward pass with autocasting.
        generated_ids = model.generate(
            pixel_values=pixel_values, max_length=args.max_token_length
        )
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        results.append({ 
            "idx": idx,
            "img": img,
            "caption": generated_caption
        })
        print(idx, img, generated_caption)

    import csv

    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["idx", "img", "caption"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/git-base",
        help="Model path",
    )

    parser.add_argument(
        "--images_dir", type=str, required=True, help="Image files to use",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=75,
        help="Maximum number of tokens to generate",
    )

    args = parser.parse_args()
    main(args)
