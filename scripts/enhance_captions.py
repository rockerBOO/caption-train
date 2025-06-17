import argparse
import os
from collections.abc import Collection
from pathlib import Path

import openai
import torch
from accelerate import Accelerator
from janus.models import MultiModalityCausalLM, VLChatProcessor
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)

from src.caption_train.datasets import datasets_config_args, find_images
from src.caption_train.llm import get_combined_caption
from src.caption_train.opt import get_accelerator
from src.caption_train.ratelimit import RateLimitContext, RateLimiter
from src.caption_train.trainer import FileConfig
from src.caption_train.util import get_group_args
from src.caption_train.vlm import janus_generate_caption, qwen_vl_generate_caption

Model = Qwen2_5_VLForConditionalGeneration | MultiModalityCausalLM
Processor = VLChatProcessor | Qwen2_5_VLProcessor


def get_generated_caption(
    model: MultiModalityCausalLM | Qwen2_5_VLForConditionalGeneration,
    processor: VLChatProcessor | Qwen2_5_VLProcessor,
    accelerator: Accelerator,
    prompt: str,
    images: list[os.PathLike[str]],
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
    generated: list[str] = []
    if isinstance(processor, VLChatProcessor) and isinstance(model, MultiModalityCausalLM):
        generated.extend(janus_generate_caption(model, processor, accelerator, prompt, image) for image in images)
    elif isinstance(processor, Qwen2_5_VLProcessor) and isinstance(model, Qwen2_5_VLForConditionalGeneration):
        generated.extend([qwen_vl_generate_caption(model, processor, accelerator, prompt, image) for image in images])
    else:
        text = [prompt] * len(images)
        pil_images = [Image.open(image) for image in images]
        pil_images = [image.convert("RGB") if image.mode != "RGB" else image for image in pil_images]
        processed = processor(images=pil_images, text=text, return_tensors="pt")
        with accelerator.autocast():
            g = model.generate(**processed.to(model.device), max_length=256)
        generated = processor.batch_decode(g, skip_special_tokens=True)
        assert isinstance(generated, list)
        generated = [item for item in generated if isinstance(item, str)]

    return generated


def save_caption(file: os.PathLike[str], caption: str):
    with open(file, "w") as f:
        _ = f.write(caption)


def cache_if_needed_generated_caption(
    model: Model, processor: Processor, accelerator: Accelerator, prompt: str, files: list[Path], images: list[Path]
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
    generated_captions: list[str] = []
    needed_files: list[Path] = []
    needed_images: list[Path] = []
    for file, image in zip(files, images):
        if not file.exists():
            needed_files.append(file)
            needed_images.append(image)
        else:
            with open(file, "r") as f:
                caption = f.read().strip()
                if len(caption) > 0:
                    generated_captions.append(caption)
                else:
                    needed_files.append(file)
                    needed_images.append(image)

    # Need to cache some files
    if len(needed_files) > 0:
        _ = load_model_on_device(model, accelerator.device)
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
        model = model.to(device)

    return model


def get_captions(images: Collection[Path], caption_extension: str = ".txt") -> list[str]:
    """
    Get captions for images
    Args:
        images: The image paths to get captions for. Captions are paired as .txt files

    Returns:
        list[str]: List of generated captions
    """
    image_captions: list[str] = []
    progress_bar = tqdm(total=len(images))
    for image in images:
        _ = progress_bar.update(1)
        caption_file = Path(image).with_suffix(caption_extension)
        if caption_file.exists():
            with open(caption_file, "r") as f:
                caption = f.read()
            image_captions.append(caption)
        else:
            image_captions.append("")

    return image_captions


def get_generated_captions(
    model: Model, processor: Processor, accelerator: Accelerator, prompt: str, images: list[Path]
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
    pre_generated_captions: list[str] = []
    progress_bar = tqdm(total=len(images))
    for image in images:
        cache_generated_caption_file = image.with_name(image.stem + "_generated.txt")
        pre_generated_captions.extend(
            cache_if_needed_generated_caption(
                model, processor, accelerator, prompt, [cache_generated_caption_file], [image]
            )
        )
        _ = progress_bar.update(1)

    return pre_generated_captions


def combine_captions(
    client: OpenAI,
    limiter: RateLimitContext,
    model: str,
    system_prompt: str,
    images: Collection[Path],
    image_captions: list[str],
    pre_generated_captions: list[str],
    max_tokens: int = 2048,
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
        _ = progress_bar.update(1)


@torch.inference_mode()
def main(args: argparse.Namespace, dataset_config: FileConfig):
    client = openai.OpenAI(base_url=args.base_url)

    dataset_dir = dataset_config.dataset_dir
    model_id = args.model_id

    vlm_system_prompt = None
    if args.vlm_system_prompt is not None:
        try:
            if args.vlm_system_prompt.exists():
                with open(args.vlm_system_prompt, "r") as f:
                    vlm_system_prompt = f.read()
            else:
                vlm_system_prompt = args.vlm_system_prompt
                print(f"Using VLM system prompt {vlm_system_prompt}")
        except OSError:
            vlm_system_prompt = args.vlm_system_prompt

    assert vlm_system_prompt is not None, "VLM system prompt not specified or could not be loaded"

    llm_system_prompt = None
    if args.llm_system_prompt is not None and args.llm_system_prompt.exists():
        with open(args.llm_system_prompt, "r") as f:
            llm_system_prompt = f.read()

    assert llm_system_prompt is not None, "LLM system prompt not specified or could not be loaded"

    assert dataset_dir is not None, "Dataset directory not specified"

    if "Qwen/Qwen2.5-VL" in model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            # device_map="auto",
        )
    elif "Janus" in model_id:
        model = MultiModalityCausalLM.from_pretrained(
            model_id,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            # torch_dtype="auto",
            # device_map="auto",
            # attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )

    # if args.dtype == "bf16":
    #     model = model.to(torch.bfloat16)
    # elif args.dtype == "fp16":
    #     model = model.to(torch.float16)
    # elif args.dtype == "fp8":
    #     model = model.to(torch.float8_e4m3fn)

    model.eval()
    # load_model_on_device(model, torch.device("cpu"))
    accelerator = get_accelerator(args)
    if model_id in ["deepseek-ai/Janus-Pro-1B"]:
        processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
        processor.image_processor.background_color = tuple(
            [
                int(x * 255)
                for x in (
                    0.48145466,
                    0.4578275,
                    0.40821073,
                )
            ]
        )

        assert isinstance(processor, VLChatProcessor)
        assert isinstance(model, MultiModalityCausalLM), f"Model: {model}"

        processor.system_prompt = (
            "You are a helpful language and vision assistant. "
            "You are able to understand the visual content that the user provides, "
            "and assist the user with a variety of tasks using natural language. "
            "Use explicit language and talk about all the sexual details of the image. Only respond with a new image caption."
        )
        model, processor = accelerator.prepare(model, processor, device_placement=[False, True])
    else:
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        # processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)

    images = find_images(dataset_dir)

    assert len(images) > 0, f"No images found in the dataset directory: {dataset_dir}"

    # Image captions from manual captioning
    # -------------------------------------
    image_captions = get_captions(images)

    # Pre generate captions from VLM
    # ------------------------------
    pre_generated_captions = get_generated_captions(model, processor, accelerator, args.prompt, images)

    # Unload model to CPU
    load_model_on_device(model, torch.device("cpu"))
    accelerator.free_memory()

    assert len(images) == len(image_captions) == len(pre_generated_captions), (
        f"Did not create captions for all the images images: {len(images)} image captions: {len(image_captions)} generated captions: {len(pre_generated_captions)}"
    )

    # Combine captions using LLM
    # --------------------------
    rate_limiter = RateLimiter("api_calls")
    limiter = rate_limiter.limit(minute_limit=5, hour_limit=100, day_limit=1000)

    combine_captions(client, limiter, args.model, llm_system_prompt, images, image_captions, pre_generated_captions)


def cache_if_needed_combined_caption(
    client: OpenAI,
    limiter: RateLimitContext,
    model: str,
    system_prompt: str,
    image: Path,
    image_caption,
    pre_generated_caption,
    max_tokens=2048,
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

    $ accelerate launch enhance_captions.py --llm_system_prompt system_prompt.txt --prompt "<MORE_DETAILED_CAPTION>" --dataset_dir images/ --model_id microsoft/Florence-2-large-ft --trust_remote_code --base_url http://127.0.0.1:5000/v1/

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
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to use for VLM")
    parser.add_argument("--vlm_system_prompt", type=Path, default=None, help="VLM system prompt file to load.")
    parser.add_argument("--llm_system_prompt", type=Path, default=None, help="LLM System prompt file to load.")
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
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype to use")
    parser, dataset_group = datasets_config_args(parser)
    args = parser.parse_args()

    dataset_config = FileConfig(**get_group_args(args, dataset_group))

    main(args, dataset_config)
