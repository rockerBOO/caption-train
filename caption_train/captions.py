from pathlib import Path
import random
import json

EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
CAPTION_EXTENSION = "txt"


def load_captions(
    dir: Path,
    captions=[],
    true_dir="",
    extensions=EXTENSIONS,
    caption_ext=CAPTION_EXTENSION,
) -> list[str]:
    found_captions = 0
    for file in dir.iterdir():
        if file.is_dir():
            print(f"found dir: {file}")
            captions = load_captions(file, captions=captions, true_dir=true_dir)
            continue

        if file.suffix.lower() not in extensions:
            continue

        # need to check for images and then get the associated .{caption_ext} file

        txt_file = file.with_name(f"{file.stem}.{caption_ext}")

        if txt_file.exists():
            with open(txt_file, "r") as f:
                file_name = str(file.relative_to(true_dir))
                caption = {
                    "file_name": file_name,
                    "text": " ".join(f.readlines()).strip(),
                }

                found_captions += 1
                # print(caption)

                captions.append(caption)
        else:
            print(f"no captions for {txt_file}")

    print(f"found_captions: {found_captions}")
    return captions


# Convert .txt captions to metadata.jsonl file for dataset
def setup_metadata(output, captions):
    captions = load_captions(output, captions, true_dir=output)

    if len(captions) == 0:
        raise ValueError("yo no captions")

    print(json.dumps(captions, indent=4))

    print("Saving captions")

    with open(Path(output) / "metadata.jsonl", "w") as f:
        # jsonl has json for each item
        for item in captions:
            f.write(json.dumps(item) + "\n")


def shuffle_caption(text, shuffle_on=", ", frozen_parts=1, dropout=0.0):
    parts = [part.strip() for part in text.split(shuffle_on)]

    # we want to keep the first part of the text, but shuffle the rest
    frozen = []

    if frozen_parts > 0:
        for i in range(frozen_parts):
            frozen.append(parts.pop(0))

    final_parts = []

    if dropout > 0.0:
        final_parts = []
        for part in parts:
            rand = random.random()
            if rand > dropout:
                final_parts.append(part)
    else:
        final_parts = parts

    random.shuffle(final_parts)

    return shuffle_on.join(frozen + final_parts)


def blend_sentences(source1, source2, num_sentences):
    """
    Blend sentences from two sources.

    Args:
        source1 (list): List of sentences from the first source.
        source2 (list): List of sentences from the second source.
        num_sentences (int): Number of sentences to generate.

    Returns:
        list: List of blended sentences.
    """

    # Check if the number of sentences is valid
    if num_sentences <= 0:
        return []

    # Initialize an empty list to store the blended sentences
    blended_sentences = []

    # Loop until we have generated the required number of sentences
    for _ in range(num_sentences):
        # Randomly select a sentence from each source
        sentence1 = random.choice(source1)
        sentence2 = random.choice(source2)

        # Randomly decide how to blend the sentences
        blend_type = random.random()

        # If blend_type is less than 0.5, append the second sentence to the first
        if blend_type < 0.5:
            blended_sentence = sentence1 + " " + sentence2
        # If blend_type is between 0.5 and 0.75, insert the second sentence in the middle of the first
        elif blend_type < 0.75:
            mid_index = len(sentence1) // 2
            blended_sentence = sentence1[:mid_index] + " " + sentence2 + " " + sentence1[mid_index:]
        # Otherwise, prepend the second sentence to the first
        else:
            blended_sentence = sentence2 + " " + sentence1

        # Add the blended sentence to the list
        blended_sentences.append(blended_sentence)

    return blended_sentences
