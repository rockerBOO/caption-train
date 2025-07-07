import pytest
import tempfile
import json
from pathlib import Path
import random

from caption_train.captions import load_captions, setup_metadata, shuffle_caption, blend_sentences


def test_load_captions():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test image and its caption
        img_path = Path(tmpdir) / "test_image.png"
        txt_path = Path(tmpdir) / "test_image.txt"

        img_path.touch()
        txt_path.write_text("A test caption")

        captions = load_captions(Path(tmpdir), true_dir=tmpdir)

        assert len(captions) == 1
        assert captions[0]["file_name"] == "test_image.png"
        assert captions[0]["text"] == "A test caption"


def test_setup_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test image and its caption
        img_path = Path(tmpdir) / "test_image.png"
        txt_path = Path(tmpdir) / "test_image.txt"

        img_path.touch()
        txt_path.write_text("A test caption")

        setup_metadata(Path(tmpdir), [])

        metadata_path = Path(tmpdir) / "metadata.jsonl"
        assert metadata_path.exists()

        with open(metadata_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            metadata = json.loads(lines[0])
            assert metadata["file_name"] == "test_image.png"
            assert metadata["text"] == "A test caption"


def test_setup_metadata_no_captions():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="yo no captions"):
            setup_metadata(Path(tmpdir), [])


def test_shuffle_caption():
    # Test basic shuffling
    original = "apple, banana, cherry, date"
    shuffled = shuffle_caption(original, shuffle_on=", ", frozen_parts=1, dropout=0.0)

    assert shuffled.startswith("apple")
    assert set(shuffled.split(", ")[1:]) == set(["banana", "cherry", "date"])


def test_shuffle_caption_dropout():
    # Seed for reproducibility
    random.seed(42)

    original = "apple, banana, cherry, date"
    shuffled = shuffle_caption(original, shuffle_on=", ", frozen_parts=1, dropout=0.5)

    assert shuffled.startswith("apple")


def test_blend_sentences():
    source1 = ["The cat", "A dog", "My bird"]
    source2 = ["is cute", "runs fast", "sings loudly"]

    # Seed for reproducibility
    random.seed(42)

    blended = blend_sentences(source1, source2, num_sentences=3)

    assert len(blended) == 3
    assert all(isinstance(sentence, str) for sentence in blended)


def test_blend_sentences_empty():
    source1 = ["The cat"]
    source2 = ["is cute"]

    blended = blend_sentences(source1, source2, num_sentences=0)

    assert blended == []
