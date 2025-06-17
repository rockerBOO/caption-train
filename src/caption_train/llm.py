from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def remove_think_tags(text: str):
    start_tag = "<think>"
    end_tag = "</think>"

    think_start_index = text.find(start_tag)
    think_end_index = text.find(end_tag) + len(end_tag)

    if think_start_index == -1 or think_end_index == -1:
        return text.strip()

    cleaned_text = text[:think_start_index].strip() + " " + text[think_end_index:].strip()

    return cleaned_text.strip()


def ask(client: OpenAI, model: str, system_prompt: str, message: str, **kwargs):
    """
    Ask question to the LLM

    Args:
        client (OpenAI): OpenAI client
        model (str): LLM model name
        system_prompt (str): System prompt
        message (str): Message
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": message,
        },
    ]

    chat_completion = client.chat.completions.create(messages=messages, model=model, stream=False, n=1, **kwargs)

    content = chat_completion.choices[0].message.content

    if content is None:
        return content

    content = remove_think_tags(content)

    return content.strip()


def ask_vision(client, model: str, system_prompt: str, message: str, base64_image: str) -> str | None:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt + "\n" + message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            },
        ],
        model="llama-3.2-11b-vision-preview",
    )

    return chat_completion.choices[0].message.content


def get_combined_caption(
    client: OpenAI, model: str, system_prompt: str, image_caption: str, pre_generated_caption: str, max_tokens=256
) -> str | None:
    """
    Combine captions using LLM
    Args:
        client: The OpenAI client
        image_caption: The caption for the image
        pre_generated_caption: The pre-generated caption

    Returns:
        str: The combined caption or None if failed
    """
    message = f"""{system_prompt}

## Important
        
{image_caption}. 

## Context 

{pre_generated_caption}"""

    combined_caption = ask(
        client,
        model,
        system_prompt,
        message=message,
        extra_body={"args": {"n_gpu_layers": 5}},
        max_tokens=max_tokens,
    )

    return combined_caption.strip() if combined_caption else None
