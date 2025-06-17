# Datasets

We need to make sure your images have a compatible dataset. We have a script to convert over image/text captioned file pairs.

Convert:

- images/img1.jpg
- images/img1.txt

To a `metadata.jsonl` using `compile_captions.py` which is compatible with 🤗Datasets.

- Find the appropriate model you want to train a LoRA on, BLIP, Florence 2, Moondream 2.
- Then we can train on that dataset.
- Also have inference scripts for each model.

### Setup dataset from text/image pairs

```
$ uv run python compile_captions.py --help
usage: compile_captions.py [-h] dataset_dir output_dir

        Create hugging face dataset compatible caption file

        Take Kohya-ss or image/text file pairs and compile it
        into a compatible file

        $ python compile_captions.py /path/to/captions/dir

        # Output metadata.jsonl to a different location
        $ python compile_captions.py /path/to/captions/dir /path/to/output_dir


positional arguments:
  dataset_dir
  output_dir

options:
  -h, --help   show this help message and exit
```

```bash
$ python compile_captions.py /path/to/captions/dir /path/to/output_dir
```
