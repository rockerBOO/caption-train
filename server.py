import argparse
from pathlib import Path

import torch
import uvicorn
from accelerate import Accelerator
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

IMAGES_DIR = None


@torch.no_grad()
def main(args):
    if args.peft_model:
        model_id = args.model_id
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=args.revision, trust_remote_code=args.trust_remote_code
        )
        model = PeftModel.from_pretrained(
            model,
            model_id=args.peft_model,
            # revision=args.revision,
            # trust_remote_code=args.trust_remote_code,
            # local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    else:
        model_id = args.model_id
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=args.revision, trust_remote_code=args.trust_remote_code
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)

    accelerator = Accelerator()
    model, processor = accelerator.prepare(model, processor)

    model.eval()

    # HTTP Server
    app = FastAPI()

    origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def read_index():
        return FileResponse("index.html")

    @app.post("/set_images_dir")
    def set_images_dir(images_dir: str):
        global IMAGES_DIR
        IMAGES_DIR = Path(images_dir)

        images = [f for f in IMAGES_DIR.glob("*.png")]
        images.sort()
        print(images)
        return {"images_dir": images_dir, "images": images}

    @app.get("/images")
    def get_images():
        images = [f for f in IMAGES_DIR.glob("*.png")]
        images.sort()
        return {"images": images}

    @app.post("/caption")
    @torch.no_grad()
    def post_item(file: UploadFile):
        text = None
        if "Peft" in model.__class__.__name__:
            if "Florence2" in model.model.__class__.__name__:
                text = args.task
        if "Florence2" in model.__class__.__name__:
            text = args.task

        with Image.open(file.file) as image:
            batch = [image]
            inputs = processor(images=batch, text=text, return_tensors="pt")

        with accelerator.autocast():
            generated_ids = model.generate(
                **inputs.to(accelerator.device),
                min_length=3,
                num_beams=args.beams,
                max_length=args.max_token_length,
            )

            generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return {"image_file": file.filename, "caption": generated_captions[0]}

    uvicorn.run(app)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Captioning API

        uv run python server.py --model_id=Salesforce/blip-image-captioning-large
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        help="Model to load from hugging face 'Salesforce/blip-image-captioning-large'",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision on Hugging Face",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code. Used in Florence 2.",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        help="PEFT (LoRA, IA3, ...) model directory for the base model",
    )
    parser.add_argument("--task", type=str, default=None, help="Task to use for the captioning")
    parser.add_argument(
        "--beams",
        type=int,
        default=3,
        help="Num beams",
    )
    parser.add_argument(
        "--save_captions",
        action="store_true",
        help="Save captions to the images next to the image",
    )
    parser.add_argument("--caption_extension", default=".txt", help="Extension to save the captions as")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=75,
        help="Maximum number of tokens to generate",
    )

    args = parser.parse_args()
    main(args)
