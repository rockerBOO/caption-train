import argparse
import glob
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from peft.auto import AutoPeftModelForCausalLM
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from caption_train.datasets import find_images

available_models = [
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large-ft",
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-base",
]


class ImageDataset(Dataset):
    def __init__(self, image_paths, task, processor):
        self.image_paths = image_paths
        self.task = task
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(text=self.task, images=image, return_tensors="pt")
        # Remove batch dimension added by processor
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.squeeze(0)
        return inputs, str(img_path)


@torch.inference_mode()
def main(args):
    accelerator = Accelerator()
    accelerator.print("Loading model")

    if args.peft_model:
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.peft_model, trust_remote_code=True, revision=args.revision
        )
    else:
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, revision=args.revision)

    if args.dtype == "bf16":
        model.to(torch.bfloat16)
    elif args.dtype == "fp16":
        model.to(torch.float16)

    model, processor = accelerator.prepare(model, processor)
    model.eval()

    accelerator.print(f"Loading images {args.images}")
    images_path = args.images

    if images_path.is_dir():
        images = list(find_images(images_path, args.recursive))
    else:
        images = [images_path]

    batch_size = args.batch_size
    task = args.task

    accelerator.print(f"Processing {len(images)} images")
    dataset = ImageDataset(images, task, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    for batch_inputs, batch_paths in tqdm(dataloader):
        with accelerator.autocast():
            process_image(model, processor, batch_inputs.to(model.device), task, batch_paths)


@torch.inference_mode()
def process_image(model, processor, batch: dict[str, torch.Tensor], task: str, images: list[Path]):
    print(f"Processing batch {batch['input_ids'].shape[0]}")
    generated_ids = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        max_new_tokens=args.max_token_length,
        do_sample=False,
        num_beams=args.num_beams,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for gen, img in zip(generated_text, images):
        print(batch["pixel_values"].shape[3], batch["pixel_values"].shape[2])
        parsed_answer = processor.post_process_generation(
            gen,
            task=task,
            image_size=(batch["pixel_values"].shape[3], batch["pixel_values"].shape[2]),
        )

        image_file = Path(img)

        print(image_file.stem)
        print(parsed_answer[task])
        process_caption(
            image_file,
            parsed_answer[task],
            save_captions=args.save_captions,
            append=args.append,
            overwrite=args.overwrite,
            caption_extension=args.caption_extension,
        )


def process_caption(
    image_file: Path, caption: str, save_captions=False, append=False, overwrite=False, caption_extension: str = ".txt"
) -> None:
    caption_file = image_file.with_name(image_file.stem + caption_extension)

    if save_captions is False:
        return

    if caption_file.is_file():
        if append is True:
            print(f"Appending to {caption_file}")
            with open(caption_file, "a", encoding="utf-8") as f:
                f.write(" " + caption)

            return

        if overwrite is False:
            print(f"Caption already exists for {str(caption_file)}")
            with open(caption_file, "r") as r:
                print(r.read())
            return

    else:
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Image captioning using Florence 2

        $ python inference.py --images /path/to/images/ --peft_model models/my_lora

        See --save_captions if producing an image/text file pair dataset.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Florence-2-base-ft",
        help="Model to load from hugging face. Default to microsoft/Florence-2-base-ft",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        help="PEFT (LoRA, IA3, ...) model directory for the base model. For transfomers models (this tool).",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory of images or an image file to caption",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for rng")
    parser.add_argument(
        "--task",
        default="<DETAILED_CAPTION>",
        choices=["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
        help="Task to run the captioning on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images to caption in the batch",
    )
    parser.add_argument(
        "--save_captions",
        default=False,
        action="store_true",
        help="Save captions to the images next to the image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite caption files with new caption. Default: We skip writing captions that already exist",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append captions to the captions that already exist or write the caption.",
    )
    parser.add_argument(
        "--caption_extension",
        default=".txt",
        help="Extension to save the captions as, .txt, .caption",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=75,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Revision of the model to load. Default: None",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images in the directory. Default: False",
    )
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams to use for beam search")
    parser.add_argument("--dtype", default=None, help="Data type to use for the model")

    args = parser.parse_args()
    main(args)
