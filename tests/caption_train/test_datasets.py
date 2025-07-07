import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers import AutoProcessor
from PIL import Image

from caption_train.datasets import (
    ImageCaptioningDataset,
    FlorenceImageTextDataset,
    Datasets,
    glob,
    find_images,
    lora_config_args,
    datasets_config_args,
)


@pytest.fixture
def mock_processor():
    # Create a mock for the entire processing workflow
    processor = MagicMock(spec=AutoProcessor)

    # Mock the tokenizer to return an object with attributes that also has a squeeze method
    def mock_tokenizer(text, **kwargs):
        class TokenizerResult:
            def __init__(self):
                self.input_ids = torch.tensor([[1, 2, 3]])
                self.attention_mask = torch.tensor([[1, 1, 1]])

            def squeeze(self):
                # Return a version with squeezed tensors
                result = TokenizerResult()
                result.input_ids = self.input_ids.squeeze()
                result.attention_mask = self.attention_mask.squeeze()
                return result

        return TokenizerResult()

    processor.tokenizer = MagicMock(side_effect=mock_tokenizer)

    # Mock the processor call to return a proper encoding dict
    def mock_call(**kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.rand(1, 3, 224, 224),
        }

    # Configure the mock to return our dict when called
    processor.side_effect = mock_call

    return processor


@pytest.fixture
def mock_image(tmp_path):
    # Create a fake image using PIL that will work in tests
    image = Image.new("RGB", (100, 100), color="red")
    mock_image_path = tmp_path / "test_image.jpg"
    image.save(mock_image_path)
    return mock_image_path


def test_image_captioning_dataset(mock_processor, monkeypatch):
    # Monkey patch shuffle_caption to return the original text
    monkeypatch.setattr("caption_train.captions.shuffle_caption", lambda text, **kwargs: text)

    # Create a mock dataset with a single item with a mock tensor
    mock_dataset = [{"image": torch.rand(3, 224, 224), "text": "Test caption"}]

    dataset = ImageCaptioningDataset(
        mock_dataset,
        mock_processor,
        max_length=10,
        caption_dropout=0.0,
        frozen_parts=0,
        shuffle_captions=False,
        transform=lambda x: x,  # Identity transform
    )

    assert len(dataset) == 1

    encoding, text = dataset[0]

    # Verify the required keys are present
    required_keys = ["input_ids", "attention_mask", "pixel_values", "label"]
    for key in required_keys:
        assert key in encoding, f"Missing key: {key}"

    assert text == "Test caption"


def test_florence_image_text_dataset(mock_processor, mock_image, monkeypatch):
    # Monkey patch shuffle_caption to return the original text
    monkeypatch.setattr("caption_train.captions.shuffle_caption", lambda text, **kwargs: text)

    # Create a mock dataset with a single item
    mock_dataset = [{"image": str(mock_image), "text": "Test caption"}]

    dataset = FlorenceImageTextDataset(
        mock_dataset,
        mock_processor,
        max_length=10,
        caption_dropout=0.0,
        frozen_parts=0,
        shuffle_captions=False,
        transform=lambda x: x,  # Identity transform
    )

    assert len(dataset) == 1

    # Mock Image.open to return a mock image
    with patch("caption_train.datasets.Image.open") as mock_image_open:
        # Mock PIL Image.open to return a mock image
        mock_pil_image = MagicMock()
        mock_image_open.return_value = mock_pil_image

        encoding, text = dataset[0]

        # Verify the required keys are present
        required_keys = ["input_ids", "attention_mask", "pixel_values", "labels"]
        for key in required_keys:
            assert key in encoding, f"Missing key: {key}"

        assert text == "Test caption"


def test_datasets():
    train_dataset = MagicMock()
    train_dataloader = MagicMock()
    accelerator = MagicMock()

    datasets = Datasets(train_dataset, train_dataloader)

    assert datasets.train_dataset == train_dataset
    assert datasets.train_dataloader == train_dataloader

    # Test accelerate method
    datasets.accelerate(accelerator)
    accelerator.prepare.assert_called_once_with(train_dataloader)


def test_glob(tmp_path):
    # Create some test files
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.jpg").touch()
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file3.png").touch()

    # Test non-recursive glob
    files = list(glob(tmp_path))
    assert len(files) == 2
    assert any(f.name == "file1.txt" for f in files)
    assert any(f.name == "file2.jpg" for f in files)

    # Test recursive glob
    files = list(glob(tmp_path, recursive=True))
    assert len(files) == 3


def test_find_images(tmp_path):
    # Create some test images and non-image files
    (tmp_path / "image1.png").touch()
    (tmp_path / "image2.jpg").touch()
    (tmp_path / "image3.jpeg").touch()
    (tmp_path / "image4.webp").touch()
    (tmp_path / "not_image.txt").touch()
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "image5.jpg").touch()

    # Test non-recursive find_images
    images = find_images(tmp_path)
    assert len(images) == 4
    assert all(img.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"] for img in images)

    # Test recursive find_images
    images = find_images(tmp_path, recursive=True)
    assert len(images) == 5


def test_lora_config_args():
    parser = MagicMock()
    parser.add_argument_group.return_value = MagicMock()

    updated_parser, group = lora_config_args(parser)

    assert updated_parser is parser
    parser.add_argument_group.assert_called_once_with("Lora Config")
    group.add_argument.assert_called_once()


def test_datasets_config_args():
    parser = MagicMock()
    parser.add_argument_group.return_value = MagicMock()

    updated_parser, group = datasets_config_args(parser)

    assert updated_parser is parser
    parser.add_argument_group.assert_called_once_with("Datasets")
    # Check that multiple arguments were added to the group
    assert group.add_argument.call_count > 5
