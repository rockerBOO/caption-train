import argparse
import os
from pathlib import Path
import openai
from openai import OpenAI
from accelerate import Accelerator

from janus.models import MultiModalityCausalLM, VLChatProcessor

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

from caption_train.opt import get_accelerator
from caption_train.llm import get_combined_caption
from caption_train.vlm import janus_generate_caption
from caption_train.ratelimit import RateLimitContext, RateLimiter


@torch.no_grad()
def get_generated_caption(
    model, processor, accelerator: Accelerator, prompt: str, images: list[os.PathLike[str]]
) -> list[str]:
    """
    Generate captions for images

    Args:
        model: The model to use for generation
        processor: The processor to use for encoding the images
        accelerator: The accelerator to use for the model
        prompt: The prompt to use for generation
        images: The images to generate captions for

    Returns:
        list[str]: List of generated captions
    """
    generated = []
    if isinstance(processor, VLChatProcessor):
        generated.extend([janus_generate_caption(model, processor, accelerator, prompt, image) for image in images])
    else:
        text = [prompt] * len(images)
        processed = processor(images=[Image.open(image) for image in images], text=text, return_tensors="pt")
        with accelerator.autocast():
            generated = model.generate(max_length=256, **processed.to(model.device))
        generated = processor.batch_decode(generated, skip_special_tokens=True)

    return generated


def save_caption(file: os.PathLike[str], caption: str):
    with open(file, "w") as f:
        f.write(caption)


def cache_if_needed_generated_caption(
    model: torch.nn.Module, processor, accelerator: Accelerator, prompt: str, files: list[Path], images: list[Path]
) -> list[str]:
    """
    Get cached generated captions for images or generate them and cache them

    Args:
        model: The model to use for generation
        processor: The processor to use for encoding the images
        accelerator: The accelerator to use for the model
        prompt: The prompt to use for generation
        files: The files to generate captions for
        images: The images to generate captions for

    Returns:
        list[str]: List of generated captions
    """
    generated_captions = []
    needed_files = []
    needed_images = []
    for file, image in zip(files, images):
        if not file.exists():
            needed_files.append(file)
            needed_images.append(image)
        else:
            with open(file, "r") as f:
                generated_captions.append(f.read())

    # Need to cache some files
    if len(needed_files) > 0:
        load_model_on_device(model, accelerator.device)
        generated = get_generated_caption(model, processor, accelerator, prompt, needed_images)
        for file, gen in zip(needed_files, generated):
            save_caption(file, gen)
            generated_captions.append(gen)

    return generated_captions


def load_model_on_device(model: torch.nn.Module, device: torch.device):
    """
    Load the model on the specified device and log if model is moved

    Args:
        model: The model to load
        device: The device to load the model on

    """
    if model.device != device:
        print(f"Loading {model.__class__.__name__} on {device}")
        model.to(device)

    return model


def get_captions(images: list[Path], caption_extension=".txt") -> list[str]:
    """
    Get captions for images
    Args:
        images: The image paths to get captions for. Captions are paired as .txt files

    Returns:
        list[str]: List of generated captions
    """
    image_captions = []
    progress_bar = tqdm(total=len(images))
    for i, image in enumerate(images):
        progress_bar.update(1)
        caption_file = Path(image).with_suffix(caption_extension)
        with open(caption_file, "r") as f:
            caption = f.read()
        image_captions.append(caption)

    return image_captions


def get_generated_captions(model, processor, accelerator, prompt, images) -> list[str]:
    """
    Generate captions for images

    Args:
        model: The model to use for generation
        processor: The processor to use for encoding the images
        accelerator: The accelerator to use for the model
        prompt: The prompt to use for generation
        images: The images to generate captions for

    Returns:
        list[str]: List of generated captions
    """
    pre_generated_captions: list[str] = []
    progress_bar = tqdm(total=len(images))
    for i, image in enumerate(images):
        cache_generated_caption_file = image.with_name(image.stem + "_generated.txt")
        pre_generated_captions.extend(
            cache_if_needed_generated_caption(
                model, processor, accelerator, prompt, [cache_generated_caption_file], [image]
            )
        )
        progress_bar.update(1)

    return pre_generated_captions


def combine_captions(
    client: OpenAI,
    limiter: RateLimitContext,
    model: str,
    system_prompt: str,
    images: list[Path],
    image_captions: list[str],
    pre_generated_captions: list[str],
    max_tokens=2048,
):
    progress_bar = tqdm(total=len(images))
    for image, image_caption, pre_generated_caption in zip(images, image_captions, pre_generated_captions):
        print("image caption")
        print("-------------")
        print(image_caption)
        print("pre generated caption")
        print("-------------")
        print(pre_generated_caption)

        combined_caption = cache_if_needed_combined_caption(
            client, limiter, model, system_prompt, image, image_caption, pre_generated_caption, max_tokens=max_tokens
        )

        print("combined caption")
        print("----------------")
        print(combined_caption)
        print("----------------")
        progress_bar.update(1)


def main(args: argparse.Namespace):
    client = openai.OpenAI(base_url=args.base_url)

    dataset_dir = args.dataset_dir
    model_id = args.model_id

    system_prompt = None
    with open(args.system_prompt, "r") as f:
        system_prompt = f.read()

    assert system_prompt is not None

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=args.revision, trust_remote_code=args.trust_remote_code
    )
    model.to(dtype=torch.bfloat16)
    model.eval()
    load_model_on_device(model, torch.device("cpu"))
    if model_id in ["deepseek-ai/Janus-Pro-1B"]:
        processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)

        assert isinstance(processor, VLChatProcessor)
        assert isinstance(model, MultiModalityCausalLM)

        processor.system_prompt = (
            "You are a helpful language and vision assistant. "
            "You are able to understand the visual content that the user provides, "
            "and assist the user with a variety of tasks using natural language. "
            "Use explicit language and talk about all the sexual details of the image. Only respond with a new image caption."
        )
    else:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)

    accelerator = get_accelerator(args)
    model, processor = accelerator.prepare(model, processor, device_placement=[False, True])

    images = (
        list(Path(dataset_dir).rglob("*.png"))
        + list(Path(dataset_dir).rglob("*.jpg"))
        + list(Path(dataset_dir).rglob("*.jpeg"))
        + list(Path(dataset_dir).rglob("*.webp"))
    )

    # Image captions from manual captioning
    # -------------------------------------
    image_captions = get_captions(images)

    # Pre generate captions from VLM
    # ------------------------------
    pre_generated_captions = get_generated_captions(model, processor, accelerator, args.prompt, images)

    # Unload model to CPU
    load_model_on_device(model, torch.device("cpu"))
    accelerator.free_memory()

    assert len(images) == len(image_captions) == len(pre_generated_captions)

    # Combine captions using LLM
    # --------------------------
    rate_limiter = RateLimiter("api_calls")
    limiter = rate_limiter.limit(minute_limit=5, hour_limit=100, day_limit=1000)

    combine_captions(client, limiter, args.model, system_prompt, images, image_captions, pre_generated_captions)


def cache_if_needed_combined_caption(
    client: OpenAI, limiter: RateLimitContext, model: str, system_prompt: str, image: Path, image_caption, pre_generated_caption, max_tokens=2048
) -> str | None:
    """
    Return the combined caption or generate combined caption if it isn't cached

    Args:
        client: The OpenAI client
        system_prompt: The system prompt to use for the LLM
        image: The image to use for the LLM
        image_caption: The caption for the image
        pre_generated_caption: The pre-generated caption

    Return:
        str | None: The combined caption or None if failed
    """
    cached_caption_file = image.with_name(image.stem + "_combined.txt")

    if cached_caption_file.exists():
        with open(cached_caption_file, "r") as f:
            combined_caption = f.read()
        return combined_caption

    with limiter:
        combined_caption = get_combined_caption(
            client, model, system_prompt, image_caption, pre_generated_caption, max_tokens=max_tokens
        )

    if combined_caption is None:
        print("Failed to get caption")
        return None

    with open(cached_caption_file, "w") as f:
        f.write(combined_caption)

    return combined_caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Enhance captions combining them with VLM and enhancing them with LLM

    $ accelerate launch enhance_captions.py --system_prompt system_prompt.txt --prompt "<MORE_DETAILED_CAPTION>" --dataset_dir images/ --model_id microsoft/Florence-2-large-ft --trust_remote_code --base_url http://127.0.0.1:5000/v1/

    set OPEN_API_KEY in your environment variables or in .env file for interacting with LLM APIs

    Flow:
    1. Load images and captions ({name}.txt)
    2. Pre-generate captions using VLM
    3. Cache pre-generated captions ({name}_generated.txt)
    4. Combine captions using LLM
    5. Cache combined captions ({name}_combined.txt)
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Dataset directory to load images from. Images can be paired with .txt files with captions.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to use for VLM")
    parser.add_argument("--system_prompt", type=str, default="system_prompt.txt", help="System prompt file to load")
    parser.add_argument("--model_id", type=str, required=True, help="VLM Model to load from hugging face")
    parser.add_argument("--revision", type=str, default="main", help="Revision to use on hugging face models")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for hugging face models")
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Base url for the LLM API. Local LLM with http://127.0.0.1:5000/v1/ is recommended.",
    )
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", help="LLM model to use")
    parser.add_argument(
        "--cache_generated_captions_to_disk", action="store_true", help="Cache generated captions to disk"
    )
    args = parser.parse_args()

    main(args)
