import base64
from collections import UserDict
from typing import Any
from collections.abc import Sequence
from os import PathLike
import os
from pathlib import Path
from io import BytesIO
import torch

from accelerate import Accelerator
from janus.models import VLChatProcessor
from janus.models.modeling_vlm import MultiModalityCausalLM
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


def encode_image(image_path: PathLike[str] | Path | Image.Image) -> str:
    """Encode an image to base64 string format.

    Args:
        image_path: Either a file path to an image or a PIL Image object

    Returns:
        str: Base64 encoded image data (without data URI prefix)
    """
    if isinstance(image_path, Image.Image):
        return base64.b64encode(image_path.tobytes()).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def image_to_data_uri(image: str | Path | PathLike[str] | Image.Image) -> str:
    """Convert an image to a data URI string with base64 encoding.

    Args:
        image: Either a file path to an image or a PIL Image object

    Returns:
        str: Data URI string in format "data:image/{format};base64,{encoded_data}"
    """
    # Handle different input types
    if isinstance(image, Image.Image):
        # Already a PIL Image object
        img = image
        img_format = img.format if img.format else "JPEG"
    else:
        # Convert to Path if it's a string or PathLike
        if isinstance(image, (str, os.PathLike)):
            image_path = Path(image)
            # Open the image
            img = Image.open(image_path)
            img_format = img.format if img.format else "JPEG"
        else:
            raise TypeError("Unsupported image type")

    # Open the image
    # Create a BytesIO object to hold the image data
    buffer = BytesIO()

    # Save the image to the BytesIO object
    img.save(buffer, format=img_format)

    # Get the bytes
    img_bytes = buffer.getvalue()

    # Encode as base64
    encoded = base64.b64encode(img_bytes).decode("utf-8")

    # Create data URI
    mime_type = f"image/{img_format.lower()}"
    data_uri = f"data:{mime_type};base64,{encoded}"

    # Close the image if we opened it (not if it was passed in)
    if not isinstance(image, Image.Image):
        img.close()

    return data_uri


#
# def qwen_vl_image_conversation(text: str, data_img: str) -> Sequence[dict[str, str | list[dict[str, str]]]]:
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": f"file://{data_img.resolve()}",
#                 },
#                 {
#                     "type": "text",
#                     "text": text,
#                 },
#             ],
#         }
#     ]
#     return messages
#
#
# def add_conversation_message(messages, prompt: str) -> Sequence[dict[str, str | list[dict[str, str]]]]:
#     messages.extend(
#         [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt,
#                     },
#                 ],
#             }
#         ]
#     )
#     return messages


def create_qwen_vl_message(text: str, image_path: str | Path | PathLike[str] | None = None) -> dict[str, Any]:
    """Create a message for Qwen VL model with optional image"""
    content = []

    if image_path is not None:
        content.append(
            {
                "type": "image",
                "image": str(image_path),
            }
        )

    content.append(
        {
            "type": "text",
            "text": text,
        }
    )

    return {
        "role": "user",
        "content": content,
    }


def qwen_vl_process_messages(
    processor: Qwen2_5_VLProcessor, messages: list[dict[str, Any]], device: torch.device
) -> UserDict:
    """Process conversation messages for Qwen VL model inference.

    Args:
        processor: Qwen2_5_VLProcessor for tokenizing and processing inputs
        messages: List of message dicts with keys:
            - "role": str - "user" or "assistant"
            - "content": List of content items with:
                * {"type": "image", "image": str} - image file path
                * {"type": "text", "text": str} - text content
        device: torch.device to place tensors on

    Returns:
        UserDict with tokenized inputs:
        - "input_ids": torch.Tensor of shape (1, seq_len)
        - "attention_mask": torch.Tensor of shape (1, seq_len)
        - "pixel_values": torch.Tensor of shape (1, num_images, C, H, W) if images present
    """
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Create inputs
    inputs = processor(
        text=[text],
        images=image_inputs or None,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
    ).to(device)

    return inputs


def qwen_vl_generate_caption(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    accelerator: Accelerator,
    prompt: str,
    image_path: str | Path | PathLike[str] | None = None,
    max_new_tokens: int = 512,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Generate caption using Qwen VL model

    Args:
        model: The Qwen VL model
        processor: The Qwen VL processor
        accelerator: Accelerator for mixed precision
        prompt: Text prompt for the model
        image_path: Optional path to an image
        max_new_tokens: Maximum number of tokens to generate
        messages: Optional list of previous messages to continue conversation

    Returns:
        Generated text response
    """
    # Create or update messages
    if messages is None:
        messages = [create_qwen_vl_message(prompt, image_path)]
    else:
        messages.append(create_qwen_vl_message(prompt, image_path))

    # Process messages to create model inputs
    inputs = qwen_vl_process_messages(processor, messages, model.device)

    # Generate output
    with torch.inference_mode():
        with accelerator.autocast():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Process output
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    assert len(output_text) == 1
    return output_text[0], messages


def qwen_vl_continue_conversation(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    accelerator: Accelerator,
    messages: list[dict[str, Any]],
    new_prompt: str,
    new_image_path: str | Path | PathLike[str] | None = None,
    max_new_tokens: int = 1024,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Continue a conversation with Qwen VL model

    Args:
        model: The Qwen VL model
        processor: The Qwen VL processor
        accelerator: Accelerator for mixed precision
        messages: Existing conversation messages
        new_prompt: New text prompt to add
        new_image_path: Optional new image to add
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated text response and updated messages
    """
    # Add new message to conversation
    messages.append(create_qwen_vl_message(new_prompt, new_image_path))

    # Process messages
    inputs = qwen_vl_process_messages(processor, messages, model.device)

    # Generate output
    with torch.inference_mode():
        with accelerator.autocast():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Process output
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Add assistant response to messages
    messages.append({"role": "assistant", "content": output_text[0]})

    return output_text[0], messages


# def qwen_vl_generate_caption(
#     model: Qwen2_5_VLForConditionalGeneration,
#     processor: Qwen2_5_VLProcessor,
#     accelerator: Accelerator,
#     prompt: str,
#     image: str | Path | PathLike[str],
#     max_new_tokens: int = 512,
# ):
#     messages = qwen_vl_image_conversation(prompt, image)
#
#     # Preparation for inference
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs or None,
#         videos=video_inputs or None,
#         padding=True,
#         return_tensors="pt",
#     ).to(model.device)
#
#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
#
#     generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     assert len(output_text) == 1
#     return output_text[0]


def janus_conversation(text: str, base64_image: str) -> Sequence[dict[str, str | list[str]]]:
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{text}",
            "images": [base64_image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return conversation


def janus_generate_caption(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    accelerator: Accelerator,
    prompt: str,
    image: PathLike[str] | str | Path | Image.Image,
) -> list[str]:
    """
    Generate captions for images using Janus

    Args:
        model: The model to use for generation
        processor: The processor to use for encoding the images
        accelerator: The accelerator to use for the model
        prompt: The prompt to use for generation
        images: The images to generate captions for

    Returns:
        list[str]: List of generated captions
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # Convert images to RGB as is required by Janus processor
    image = image.convert("RGB")

    assert isinstance(processor, VLChatProcessor)

    # ensure the background color is set due to errors
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

    conversation = janus_conversation(prompt, encode_image(image))

    processed = processor(images=[image], conversations=conversation, return_tensors="pt", force_batchify=True).to(
        model.device
    )

    # run the model to get the response
    with accelerator.autocast():
        # processed.images_seq_mask = processed.images_seq_mask.long()
        inputs_embeds = model.prepare_inputs_embeds(**processed)
        generated = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=processed.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
        )
    generated = processor.tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)

    return generated
