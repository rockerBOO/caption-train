from pathlib import Path
import random
import json


def load_captions(dir: Path, captions=[], true_dir="") -> list[str]:
    found_captions = 0
    for file in dir.iterdir():
        if file.is_dir():
            print(f"found dir: {file}")
            captions = load_captions(file, captions=captions, true_dir=true_dir)
            continue

        if file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            continue

        # need to check for images and then get the associated .txt file

        txt_file = file.with_name(f"{file.stem}.txt")

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

    # print(json.dumps(captions, indent=4))

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
