import json
import argparse
from pathlib import Path
from caption_train.captions import load_captions

from datasets import load_dataset


def main(args):
    dataset_dir = Path(args.dataset_dir)
    output = (
        Path(args.output_dir) if args.output_dir is not None else dataset_dir
    )
    captions = load_captions(dataset_dir, captions=[], true_dir=output)

    if len(captions) == 0:
        raise ValueError("yo no captions")

    print(json.dumps(captions, indent=4))

    print("Saving captions")

    with open(output / "metadata.jsonl", "w") as f:
        # jsonl has json for each item
        for item in captions:
            f.write(json.dumps(item) + "\n")

    ds = load_dataset(
        "imagefolder",
        data_dir=args.dataset_dir,
        split="train",
    )

    print("Successfully created captions and tested with datasets library")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="""
        Create hugging face dataset compatible caption file

        Take Kohya-ss or image/text file pairs and compile it
        into a compatible file

        python compile_captions.py /path/to/captions/dir

        # Output metadata.jsonl to a different location
        python compile_captions.py /path/to/captions/dir /path/to/output_dir
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    arg_parser.add_argument("dataset_dir")
    arg_parser.add_argument("output_dir", default=None)
    args = arg_parser.parse_args()
    main(args)
