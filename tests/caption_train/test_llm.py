from unittest.mock import MagicMock
from caption_train.llm import remove_think_tags, ask, ask_vision, get_combined_caption


def test_remove_think_tags():
    # Test with think tags
    text = "Some text <think>hidden think text</think> rest of text"
    result = remove_think_tags(text)
    assert result == "Some text rest of text"

    # Test without think tags
    text = "Some text without tags"
    result = remove_think_tags(text)
    assert result == text

    # Test with think tags at the beginning
    text = "<think>hidden think text</think> rest of text"
    result = remove_think_tags(text)
    assert result == "rest of text"

    # Test with think tags at the end
    text = "Some text <think>hidden think text</think>"
    result = remove_think_tags(text)
    assert result == "Some text"

    # Test with empty text
    text = ""
    result = remove_think_tags(text)
    assert result == ""


def test_ask():
    # Create a mock that mimics the OpenAI client structure
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "<think>Hidden text</think>Response text"

    # Setup the mock to return our predefined response
    mock_client.chat.completions.create.return_value = mock_response

    # Call the function
    result = ask(mock_client, "gpt-3.5-turbo", "You are a helpful assistant", "What is the capital of France?")

    # Verify the result
    assert result == "Response text"

    # Verify the client method was called with correct arguments
    mock_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-3.5-turbo"
    assert len(kwargs["messages"]) == 2


def test_ask_with_none_content():
    # Create a mock that mimics the OpenAI client structure
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None

    # Setup the mock to return our predefined response
    mock_client.chat.completions.create.return_value = mock_response

    # Call the function
    result = ask(mock_client, "gpt-3.5-turbo", "You are a helpful assistant", "What is the capital of France?")

    # Verify the result
    assert result is None


def test_ask_vision():
    # Create a mock that mimics the OpenAI client structure
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "A description of the image"

    # Setup the mock to return our predefined response
    mock_client.chat.completions.create.return_value = mock_response

    # Call the function
    result = ask_vision(
        mock_client, "llama-3.2-11b-vision-preview", "Describe the image", "What do you see?", "base64_encoded_image"
    )

    # Verify the result
    assert result == "A description of the image"

    # Verify the client method was called with correct arguments
    mock_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_client.chat.completions.create.call_args
    assert len(kwargs["messages"][0]["content"]) == 2
    assert kwargs["messages"][0]["content"][1]["type"] == "image_url"


def test_get_combined_caption():
    # Create a mock that mimics the OpenAI client structure
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Combined caption text"

    # Setup the mock to return our predefined response
    mock_client.chat.completions.create.return_value = mock_response

    # Call the function
    result = get_combined_caption(
        mock_client, "gpt-3.5-turbo", "You are a helpful assistant", "Image caption", "Pre-generated caption"
    )

    # Verify the result
    assert result == "Combined caption text"

    # Verify the client method was called with correct arguments
    mock_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_client.chat.completions.create.call_args
    assert "max_tokens" in kwargs
    assert "extra_body" in kwargs

    # Check the content of the messages
    assert len(kwargs["messages"]) > 0
    assert kwargs["messages"][0]["content"] is not None
