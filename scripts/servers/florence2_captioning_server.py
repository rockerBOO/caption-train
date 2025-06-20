import argparse
from pathlib import Path
import gradio as gr

import torch
from accelerate import Accelerator
from peft import AutoPeftModelForCausalLM
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

available_models = [
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large-ft",
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-base",
]


@torch.inference_mode()
def main(args):
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    accelerator = Accelerator()

    if args.peft_model is None:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.peft_model,
            trust_remote_code=True,
        )

    model, processor = accelerator.prepare(model, processor)
    model.eval()

    def greet(image):
        batch_size = args.batch_size

        task = args.task

        images = [image]

        for i in range(0, len(images), batch_size):
            batch = [Image.open(img).convert("RGB") for img in images[i : i + batch_size]]

            inputs = processor(text=[task] * len(batch), images=batch, return_tensors="pt")

            with accelerator.autocast():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"].to(accelerator.device),
                    pixel_values=inputs["pixel_values"].to(accelerator.device),
                    max_new_tokens=args.max_token_length,
                    do_sample=False,
                    num_beams=3,
                )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)

            for gen, img in zip(generated_text, images[i : i + batch_size]):
                parsed_answer = processor.post_process_generation(
                    gen,
                    task=task,
                    image_size=(batch[0].width, batch[0].height),
                )

                image_file = Path(img)

                print(image_file.stem)
                print(parsed_answer[task])

                # caption_file = image_file.with_name(
                #     image_file.stem + args.caption_extension
                # )
                #
                # if caption_file.is_file():
                #     if args.overwrite is False:
                #         # print(f"Caption already exists for {str(caption_file)}")
                #         # with open(caption_file, "r") as r:
                #         #     print(r.read())
                #         continue
                #
                # with open(caption_file, "w", encoding="utf-8") as f:
                #     f.write(parsed_answer[task])

    demo = gr.Interface(
        fn=greet,
        inputs=["image"],
        outputs=["text"],
    )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Florence 2 inference server - WIP
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--base_model",
        type=str,
        choices=available_models,
        default="microsoft/Florence-2-base-ft",
        help="Model to load from hugging face 'microsoft/Florence-2-base-ft'",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        help="PEFT (LoRA, IA3, ...) model directory for the base model. For transfomers models (this tool).",
    )
    parser.add_argument("--seed", type=int, help="Seed for rng")
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
        action="store_true",
        help="Save captions to the images next to the image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite caption files with new caption. Default: We skip writing captions that already exist",
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

    args = parser.parse_args()
    main(args)
