# Captions

For your dataset you'll want to caption your images. We support `metadata.jsonl` files currently which have your image/caption pairs.

```jsonl
{"file_name": "image.jpg", "text": "Caption to this image"}
{"file_name": "image2.jpg", "text": "Caption to this image"}
```

We have a script to do this for you if you have image/caption pairs.

- `images/img1.jpg`
- `images/img1.txt`

See `python compile_captions.py --help`.

Then you can train using this dataset. The `metadata.jsonl` gets placed in the directory when doing this process. For example: `images/metadata.jsonl`.

You should compile your captions every time you update the dataset. It is a fairly fast process.
