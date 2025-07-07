import pytest
import torch
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

from caption_train.inference.sampling import (
    sample_captions,
    sample_with_prompts,
    print_sample_comparison,
    evaluate_sample_batch,
)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    return model


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.batch_decode.return_value = ["Generated caption"]
    return processor


@pytest.fixture
def mock_accelerator():
    accelerator = MagicMock()

    @contextmanager
    def mock_autocast():
        yield

    accelerator.autocast = mock_autocast
    return accelerator


@pytest.fixture
def sample_batch():
    return {
        "pixel_values": torch.rand(1, 3, 224, 224),
        "input_ids": torch.tensor([[1, 2, 3]]),
        "text": ["Original caption"],
    }


def test_sample_captions_basic(mock_model, mock_processor, mock_accelerator, sample_batch):
    """Test basic caption sampling."""
    captions = sample_captions(mock_model, mock_processor, mock_accelerator, sample_batch)
    assert captions == ["Generated caption"], f"Unexpected captions: {captions}"  # noqa: F841  # linter-ignore-unused = True

    # Verify model.generate was called
    mock_model.generate.assert_called_once()

    # Verify processor.batch_decode was called
    mock_processor.batch_decode.assert_called_once()

    # Verify return value
    assert captions == ["Generated caption"]


def test_sample_captions_with_parameters(mock_model, mock_processor, mock_accelerator, sample_batch):
    """Test caption sampling with custom parameters."""
    captions = sample_captions(
        mock_model,
        mock_processor,
        mock_accelerator,
        sample_batch,
        max_length=128,
        num_beams=3,
        min_length=5,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    assert captions == ["Generated caption"], f"Unexpected captions: {captions}"  # noqa: F841  # linter-ignore-unused = True

    # Verify generate was called with correct parameters
    call_args = mock_model.generate.call_args
    assert call_args[1]["max_length"] == 128
    assert call_args[1]["num_beams"] == 3
    assert call_args[1]["min_length"] == 5
    assert call_args[1]["do_sample"] is True
    assert call_args[1]["temperature"] == 0.8
    assert call_args[1]["top_p"] == 0.9


def test_sample_captions_without_input_ids(mock_model, mock_processor, mock_accelerator):
    """Test caption sampling without input_ids in batch."""
    batch = {"pixel_values": torch.rand(1, 3, 224, 224)}

    captions = sample_captions(mock_model, mock_processor, mock_accelerator, batch)
    assert captions == ["Generated caption"], f"Unexpected captions: {captions}"  # noqa: F841  # linter-ignore-unused = True

    # Verify generate was called without input_ids
    call_args = mock_model.generate.call_args
    assert "input_ids" not in call_args[1]
    assert "pixel_values" in call_args[1]


def test_sample_with_prompts_no_prompts(mock_model, mock_processor, mock_accelerator):
    """Test sampling with prompts when no prompts provided."""
    images = torch.rand(2, 3, 224, 224)

    mock_processor.return_value = {"pixel_values": images, "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]])}

    captions = sample_with_prompts(mock_model, mock_processor, mock_accelerator, images)
    assert captions == ["Generated caption"], f"Unexpected captions: {captions}"  # noqa: F841  # linter-ignore-unused = True

    # Verify processor was called for images only
    mock_processor.assert_called_once()
    call_args = mock_processor.call_args
    assert torch.equal(call_args[1]["images"], images)
    assert "text" not in call_args[1]

    assert captions == ["Generated caption"]


def test_sample_with_prompts_with_prompts(mock_model, mock_processor, mock_accelerator):
    """Test sampling with text prompts."""
    images = torch.rand(2, 3, 224, 224)
    prompts = ["Describe this image", "What do you see?"]

    mock_processor.return_value = {"pixel_values": images, "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]])}

    captions = sample_with_prompts(mock_model, mock_processor, mock_accelerator, images, prompts)
    assert captions == ["Generated caption"], f"Unexpected captions: {captions}"  # noqa: F841  # linter-ignore-unused = True

    # Verify processor was called with images and prompts
    call_args = mock_processor.call_args
    assert torch.equal(call_args[1]["images"], images)
    assert call_args[1]["text"] == prompts
    assert call_args[1]["padding"] is True

    assert captions == ["Generated caption"]


def test_sample_with_prompts_custom_generation_kwargs(mock_model, mock_processor, mock_accelerator):
    """Test sampling with custom generation parameters."""
    images = torch.rand(1, 3, 224, 224)

    mock_processor.return_value = {
        "pixel_values": images,
    }

    captions = sample_with_prompts(
        mock_model, mock_processor, mock_accelerator, images, max_length=128, num_beams=5, do_sample=True
    )
    assert captions == ["Generated caption"], f"Unexpected captions: {captions}"  # noqa: F841  # linter-ignore-unused = True

    # Verify generate was called with custom parameters
    call_args = mock_model.generate.call_args
    assert call_args[1]["max_length"] == 128
    assert call_args[1]["num_beams"] == 5
    assert call_args[1]["do_sample"] is True


def test_print_sample_comparison(capsys):
    """Test printing caption comparison."""
    original = ["Original caption 1", "Original caption 2", "Original caption 3"]
    generated = ["Generated caption 1", "Generated caption 2", "Generated caption 3"]

    print_sample_comparison(original, generated, max_examples=2)

    captured = capsys.readouterr()
    assert "CAPTION COMPARISON" in captured.out
    assert "Example 1:" in captured.out
    assert "Example 2:" in captured.out
    assert "Example 3:" not in captured.out
    assert "Original caption 1" in captured.out
    assert "Generated caption 1" in captured.out


def test_print_sample_comparison_empty_lists(capsys):
    """Test printing comparison with empty lists."""
    print_sample_comparison([], [])

    captured = capsys.readouterr()
    assert "CAPTION COMPARISON" in captured.out
    assert "Example 1:" not in captured.out


def test_evaluate_sample_batch_with_text(mock_model, mock_processor, mock_accelerator, sample_batch, capsys):
    """Test evaluating sample batch with text field."""
    with patch("caption_train.inference.sampling.sample_captions") as mock_sample_captions:
        mock_sample_captions.return_value = ["Generated caption"]

        original, generated = evaluate_sample_batch(mock_model, mock_processor, mock_accelerator, sample_batch)

        # Verify sample_captions was called
        mock_sample_captions.assert_called_once()

        # Verify return values
        assert original == ["Original caption"]
        assert generated == ["Generated caption"]

        # Verify comparison was printed
        captured = capsys.readouterr()
        assert "CAPTION COMPARISON" in captured.out


def test_evaluate_sample_batch_with_input_ids(mock_model, mock_processor, mock_accelerator):
    """Test evaluating sample batch with input_ids field."""
    batch = {"pixel_values": torch.rand(1, 3, 224, 224), "input_ids": torch.tensor([[1, 2, 3]])}

    mock_processor.batch_decode.return_value = ["Decoded original"]

    with patch("caption_train.inference.sampling.sample_captions") as mock_sample_captions:
        mock_sample_captions.return_value = ["Generated caption"]

        original, generated = evaluate_sample_batch(
            mock_model, mock_processor, mock_accelerator, batch, print_comparison=False
        )

        # Verify original captions were decoded from input_ids
        assert original == ["Decoded original"]
        assert generated == ["Generated caption"]


def test_evaluate_sample_batch_no_original_captions(mock_model, mock_processor, mock_accelerator, capsys):
    """Test evaluating sample batch without original captions."""
    batch = {"pixel_values": torch.rand(1, 3, 224, 224)}

    with patch("caption_train.inference.sampling.sample_captions") as mock_sample_captions:
        mock_sample_captions.return_value = ["Generated caption"]

        original, generated = evaluate_sample_batch(mock_model, mock_processor, mock_accelerator, batch)

        # Verify return values
        assert original == []
        assert generated == ["Generated caption"]

        # Verify generated captions were printed
        captured = capsys.readouterr()
        assert "Generated captions:" in captured.out
        assert "1: Generated caption" in captured.out


def test_evaluate_sample_batch_custom_generation_kwargs(mock_model, mock_processor, mock_accelerator, sample_batch):
    """Test evaluating sample batch with custom generation parameters."""
    generation_kwargs = {"max_length": 128, "num_beams": 3}

    with patch("caption_train.inference.sampling.sample_captions") as mock_sample_captions:
        mock_sample_captions.return_value = ["Generated caption"]

        evaluate_sample_batch(
            mock_model, mock_processor, mock_accelerator, sample_batch, generation_kwargs=generation_kwargs
        )

        # Verify sample_captions was called with custom kwargs
        call_args = mock_sample_captions.call_args
        assert call_args[1]["max_length"] == 128
        assert call_args[1]["num_beams"] == 3


def test_evaluate_sample_batch_no_print(mock_model, mock_processor, mock_accelerator, sample_batch, capsys):
    """Test evaluating sample batch without printing."""
    with patch("caption_train.inference.sampling.sample_captions") as mock_sample_captions:
        mock_sample_captions.return_value = ["Generated caption"]

        evaluate_sample_batch(mock_model, mock_processor, mock_accelerator, sample_batch, print_comparison=False)

        # Verify nothing was printed
        captured = capsys.readouterr()
        assert captured.out == ""
