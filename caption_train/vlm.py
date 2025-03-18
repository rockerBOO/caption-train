from pathlib import Path
from PIL import Image
import base64

from accelerate import Accelerator
from janus.models import VLChatProcessor
from janus.models.modeling_vlm import MultiModalityCausalLM


# Function to encode the image
def encode_image(image_path):
    if isinstance(image_path, Image.Image):
        return base64.b64encode(image_path.tobytes()).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def vlm_conversation(text: str, base64_image: str) -> list:
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
    image: str | Path | Image.Image,
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
    processor.image_processor.background_color = tuple([int(x * 255) for x in (
        0.48145466,
        0.4578275,
        0.40821073,
    )])

    conversation = vlm_conversation(prompt, encode_image(image))

    processed = processor(images=[image], conversations=conversation, return_tensors="pt", force_batchify=True).to(model.device)

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
