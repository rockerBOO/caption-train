"""Sampling and inference utilities for trained models."""

import torch
from typing import Any
from accelerate import Accelerator
from transformers import AutoProcessor


@torch.no_grad()
def sample_captions(
    model: torch.nn.Module,
    processor: AutoProcessor,
    accelerator: Accelerator,
    batch: dict[str, torch.Tensor],
    max_length: int = 256,
    num_beams: int = 1,
    min_length: int = 3,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[str]:
    """Generate captions for a batch of images.

    Args:
        model: Trained model for caption generation
        processor: Processor for decoding outputs
        accelerator: Accelerator for mixed precision and device management
        batch: Batch containing pixel_values and other input data
        max_length: Maximum length of generated captions
        num_beams: Number of beams for beam search (1 for greedy)
        min_length: Minimum length of generated captions
        do_sample: Whether to use sampling instead of greedy/beam search
        temperature: Sampling temperature (only used if do_sample=True)
        top_p: Top-p sampling parameter (only used if do_sample=True)

    Returns:
        List of generated caption strings
    """
    device = model.device
    dtype = getattr(model, "dtype", torch.float32)

    with accelerator.autocast(), torch.inference_mode():
        # Generate captions
        generation_kwargs = {
            "pixel_values": batch["pixel_values"].to(device, dtype=dtype),
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
        }

        # Add sampling parameters if using sampling
        if do_sample:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        # Handle models that need input_ids for generation
        if "input_ids" in batch:
            generation_kwargs["input_ids"] = batch["input_ids"].to(device)

        generated_ids = model.generate(**generation_kwargs)

        # Decode generated captions
        generated_captions = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    return generated_captions


@torch.no_grad()
def sample_with_prompts(
    model: torch.nn.Module,
    processor: AutoProcessor,
    accelerator: Accelerator,
    images: torch.Tensor,
    prompts: list[str] | None = None,
    max_length: int = 256,
    **generation_kwargs,
) -> list[str]:
    """Generate captions with optional text prompts.

    Args:
        model: Trained model for caption generation
        processor: Processor for encoding inputs and decoding outputs
        accelerator: Accelerator for mixed precision
        images: Batch of image tensors
        prompts: Optional text prompts to guide generation
        max_length: Maximum length of generated captions
        **generation_kwargs: Additional arguments for model.generate()

    Returns:
        List of generated caption strings
    """
    device = model.device
    dtype = getattr(model, "dtype", torch.float32)

    with accelerator.autocast(), torch.inference_mode():
        # Prepare inputs
        if prompts is not None:
            # Process images and prompts together
            inputs = processor(
                images=images,
                text=prompts,
                return_tensors="pt",
                padding=True,
            )
        else:
            # Process only images
            inputs = processor(
                images=images,
                return_tensors="pt",
            )

        # Move to device and convert dtype for pixel values
        for key, value in inputs.items():
            if key == "pixel_values":
                inputs[key] = value.to(device, dtype=dtype)
            else:
                inputs[key] = value.to(device)

        # Generate captions
        generated_ids = model.generate(**inputs, max_length=max_length, **generation_kwargs)

        # Decode generated captions
        generated_captions = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    return generated_captions


def print_sample_comparison(
    original_captions: list[str],
    generated_captions: list[str],
    max_examples: int = 5,
) -> None:
    """Print comparison between original and generated captions.

    Args:
        original_captions: Original caption texts
        generated_captions: Generated caption texts
        max_examples: Maximum number of examples to print
    """
    print("\n" + "=" * 50)
    print("CAPTION COMPARISON")
    print("=" * 50)

    for i, (orig, gen) in enumerate(zip(original_captions, generated_captions)):
        if i >= max_examples:
            break

        print(f"\nExample {i + 1}:")
        print(f"Original: {orig}")
        print(f"Generated: {gen}")

    print("=" * 50 + "\n")


@torch.no_grad()
def evaluate_sample_batch(
    model: torch.nn.Module,
    processor: AutoProcessor,
    accelerator: Accelerator,
    batch: dict[str, torch.Tensor],
    print_comparison: bool = True,
    generation_kwargs: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Evaluate model on a sample batch and optionally print results.

    Args:
        model: Trained model
        processor: Processor for the model
        accelerator: Accelerator instance
        batch: Batch of data including pixel_values and text
        print_comparison: Whether to print caption comparison
        generation_kwargs: Additional generation parameters

    Returns:
        Tuple of (original_captions, generated_captions)
    """
    generation_kwargs = generation_kwargs or {}

    # Get original captions if available
    original_captions = []
    if "text" in batch:
        original_captions = batch["text"]
    elif "input_ids" in batch:
        original_captions = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)

    # Generate new captions
    generated_captions = sample_captions(model, processor, accelerator, batch, **generation_kwargs)

    # Print comparison if requested
    if print_comparison and original_captions:
        print_sample_comparison(original_captions, generated_captions)
    elif print_comparison:
        print("\nGenerated captions:")
        for i, caption in enumerate(generated_captions):
            print(f"{i + 1}: {caption}")

    return original_captions, generated_captions
