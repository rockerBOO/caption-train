import argparse
from pathlib import Path
from typing import Any
from tqdm import tqdm
import torch
from accelerate import Accelerator
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from caption_train.datasets import datasets_config_args
from caption_train.opt import get_accelerator
from caption_train.trainer import FileConfig
from caption_train.util import get_group_args
from caption_train.vlm import qwen_vl_generate_caption


@torch.inference_mode()
def main(args: argparse.Namespace, dataset_config: FileConfig):
    dataset_dir = dataset_config.dataset_dir
    assert dataset_dir is not None
    # model_id = args.model_id

    # Load system prompts
    # vlm_system_prompt = None
    # if args.vlm_system_prompt is not None:
    #     try:
    #         if args.vlm_system_prompt.exists():
    #             with open(args.vlm_system_prompt, "r") as f:
    #                 vlm_system_prompt = f.read()
    #         else:
    #             vlm_system_prompt = args.vlm_system_prompt
    #             print(f"Using VLM system prompt {vlm_system_prompt}")
    #     except OSError:
    #         vlm_system_prompt = args.vlm_system_prompt
    # assert vlm_system_prompt is not None, "VLM system prompt not specified or could not be loaded"
    # parser = argparse.ArgumentParser(description="Improve image captions using Qwen2.5VL")
    # parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    # parser.add_argument("--caption", type=str, required=True, help="Initial caption for the image")
    # args = parser.parse_args()

    # Setup accelerator
    accelerator = Accelerator()

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Setup accelerator
    accelerator = get_accelerator(args)
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    print(model_id)
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    model = accelerator.prepare(model)

    # # Load image
    # image_path = Path(args.image)
    # if not image_path.exists():
    #     raise FileNotFoundError(f"Image file not found: {image_path}")
    #
    # # Load caption from file
    # caption_file_path = Path(args.caption_file)
    # if not caption_file_path.exists():
    #     raise FileNotFoundError(f"Caption file not found: {caption_file_path}")
    #
    # with open(caption_file_path, "r", encoding="utf-8") as f:
    #     initial_caption = f.read().strip()

    # print([img.stem.lower() for img in dataset_dir.iterdir()])
    images = [
        img
        for img in dataset_dir.iterdir()
        if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".avif"]
    ]
    print(images)
    for image_file in tqdm(images, total=len(images)):
        caption_file = image_file.with_name(f"{image_file.stem}_combined.txt")
        output_caption_file = image_file.with_name(f"{image_file.stem}_combined_improved.txt")
        if not caption_file.exists():
            print(f"Caption file not found for {image_file.name}, skipping...")
            continue

        if output_caption_file.exists():
            print(f"Output caption file already exists for {image_file.name}, skipping...")
            continue

        # Read original caption
        with open(caption_file, "r", encoding="utf-8") as f:
            initial_caption = f.read().strip()

        print(f"\nProcessing: {image_file}")
        print(f"Original Caption: {initial_caption}")

        improved = improve_caption(model, processor, accelerator, image_file, initial_caption)

        new_caption = improved.get("new_caption", None)

        assert new_caption is not None

        print("--- Analysis ---\n")
        print(f"{improved.get('analysis')}\n\n")
        # print("--- Improvement Plan ---\n")
        # print(f"{improved.get('plan')}\n\n")
        print("Improved Caption:")
        print(new_caption)

        def remove_outer_quotes(text):
            if text and len(text) >= 2:
                # Check if text starts and ends with either single or double quotes
                if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
                    return text[1:-1]
            return text

        # Save results to a text file
        with open(output_caption_file, "w") as f:
            f.write(remove_outer_quotes(new_caption))

        print(f"\nResults saved to: {output_caption_file}")


def improve_caption(model, processor, accelerator, image_path, initial_caption) -> dict[str, Any]:
    # Start conversation with analysis
    analysis_prompt = f"""
    I want you to analyze this image and the caption: "{initial_caption}"
    
    Please provide a detailed analysis of:
    - What key objects, people, actions, or settings are present in the image but missing from the caption
    - Any inaccuracies in the caption (incorrect descriptions, relationships, or attributes)
    - Whether the caption captures the main focal point or primary subject
    - How well important visual details (colors, positioning, expressions, etc.) are represented
    - If the caption's tone and style match the image content
    
    Format your analysis as a clear list of observations. In the analysis be as concise as possible.

    Create a systematic plan to improve the caption:
    - Prioritize corrections for factual inaccuracies first
    - Outline specific additions needed to capture missing key elements
    - Determine if certain details should be removed or de-emphasized
    - Plan how to maintain the original caption's strengths
    - Consider how to preserve the intended context while making improvements
    - Structure changes to maintain natural language flow
    """

    print("Analysis...")
    analysis, messages = qwen_vl_generate_caption(
        model, processor, accelerator, analysis_prompt, image_path, max_new_tokens=2048
    )

    # # Continue conversation with planning
    # planning_prompt = """
    # Please create a systematic plan to improve the caption:
    # - Prioritize corrections for factual inaccuracies first
    # - Outline specific additions needed to capture missing key elements
    # - Determine if certain details should be removed or de-emphasized
    # - Plan how to maintain the original caption's strengths
    # - Consider how to preserve the intended context while making improvements
    # - Structure changes to maintain natural language flow
    #
    # Format your plan as a clear, step-by-step list. Do not repeat the analysis in the plan. Be concise in the plan to the details needed.
    # """
    #
    # print("Planning...")
    # plan, messages = qwen_vl_continue_conversation(model, processor, accelerator, messages, planning_prompt)

    # Final step - generate improved caption
    generation_prompt = f"""

    We want to improve the caption for this image

    Initial caption: 

    {initial_caption}

    Analysis and plan for improvement:
    
    {analysis}

    Based on this analysis and plan, please create a new, improved caption for this image. 
    The caption should:
    * Incorporate all identified missing key elements
    * Correct any inaccuracies from the original
    * Ensure the main subject/focus is properly emphasized
    * Include relevant contextual details
    * Use clear, concise language that accurately reflects the image
    
    Provide only the new caption without explanation.
    """

    new_caption, messages = qwen_vl_generate_caption(
        model, processor, accelerator, generation_prompt, image_path, max_new_tokens=512
    )

    return {"analysis": analysis, "new_caption": new_caption, "messages": messages}


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
    # parser.add_argument("--model_id", type=str, required=True, help="VLM Model to load from hugging face")
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
