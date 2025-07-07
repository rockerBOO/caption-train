import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock, patch

from caption_train.vlm import (
    encode_image,
    image_to_data_uri,
    create_qwen_vl_message,
    qwen_vl_process_messages,
    janus_conversation,
)


@pytest.fixture
def test_image():
    # Create a simple test image
    image = Image.new("RGB", (100, 100), color="red")
    return image


def test_encode_image(test_image, tmp_path):
    # Test with PIL Image
    encoded_pil = encode_image(test_image)
    assert isinstance(encoded_pil, str)

    # Test with file path
    test_image_path = tmp_path / "test_image.png"
    test_image.save(test_image_path)

    encoded_path = encode_image(test_image_path)
    assert isinstance(encoded_path, str)
    assert encoded_path != encoded_pil


def test_image_to_data_uri(test_image, tmp_path):
    # Test with PIL Image
    data_uri_pil = image_to_data_uri(test_image)
    assert isinstance(data_uri_pil, str)
    assert data_uri_pil.startswith("data:image/")
    assert data_uri_pil.startswith(
        "data:image/jpeg;base64," if test_image.format is None else f"data:image/{test_image.format.lower()};base64,"
    )

    # Test with file path
    test_image_path = tmp_path / "test_image.png"
    test_image.save(test_image_path)

    data_uri_path = image_to_data_uri(test_image_path)
    assert isinstance(data_uri_path, str)
    assert data_uri_path.startswith("data:image/png;base64,")


def test_create_qwen_vl_message(tmp_path):
    # Test without image
    message_text = create_qwen_vl_message("Test prompt")
    assert message_text["role"] == "user"
    assert len(message_text["content"]) == 1
    assert message_text["content"][0]["type"] == "text"
    assert message_text["content"][0]["text"] == "Test prompt"

    # Test with image
    test_image_path = tmp_path / "test_image.png"
    Image.new("RGB", (100, 100), color="red").save(test_image_path)

    message_with_image = create_qwen_vl_message("Test prompt", test_image_path)
    assert message_with_image["role"] == "user"
    assert len(message_with_image["content"]) == 2
    assert message_with_image["content"][0]["type"] == "image"
    assert message_with_image["content"][1]["type"] == "text"


def test_qwen_vl_process_messages():
    # Create mock objects
    device = torch.device("cpu")
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "Processed text"

    # Mock process_vision_info to return empty lists
    with patch("caption_train.vlm.process_vision_info", return_value=(None, None)) as mock_process_vision:
        # Prepare messages
        messages = [{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}]

        # Create processor mock
        mock_processor.return_value.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        # Call the function
        qwen_vl_process_messages(mock_processor, messages, device)

        # Verify function calls
        mock_processor.apply_chat_template.assert_called_once()
        mock_process_vision.assert_called_once_with(messages)
        mock_processor.assert_called_once()


# Remaining tests would require extensive mocking and setup of specific models and processors
# These would typically be integration or end-to-end tests
# Here we'll just add basic sanity checks
def test_janus_conversation(test_image):
    # Encode test image
    base64_image = encode_image(test_image)

    conversation = janus_conversation("Test prompt", base64_image)

    assert len(conversation) == 2
    assert conversation[0]["role"] == "<|User|>"
    assert conversation[0]["content"].endswith("Test prompt")
    assert conversation[0]["images"] == [base64_image]
    assert conversation[1]["role"] == "<|Assistant|>"
