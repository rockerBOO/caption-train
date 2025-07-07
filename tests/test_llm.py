from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
import pytest
from unittest.mock import MagicMock, patch
import base64

from caption_train.llm import remove_think_tags, ask, ask_vision, get_combined_caption


def test_remove_think_tags():
    text = "Some text <think>thinking</think> more text"
    result = remove_think_tags(text)
    assert result == "Some text more text"

    text = "<think>hidden content</think>visible content"
    result = remove_think_tags(text)
    assert result == "visible content"


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_completion():
    return ChatCompletion(
        id="123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="<think>thinking</think>response content",
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=15, prompt_tokens=25, total_tokens=40),
    )


@pytest.fixture
def mock_vision_completion():
    return ChatCompletion(
        id="vision123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="vision model response",
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="llama-3.2-11b-vision-preview",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=15, prompt_tokens=25, total_tokens=40),
    )


def test_ask(mock_client, mock_completion):
    mock_client.chat.completions.create.return_value = mock_completion

    result = ask(mock_client, "gpt-4", "system prompt", "user message")

    mock_client.chat.completions.create.assert_called_once()
    assert result == "response content"


def test_ask_with_none_content(mock_client):
    completion = MagicMock()
    completion.choices[0].message.content = None
    mock_client.chat.completions.create.return_value = completion

    result = ask(mock_client, "gpt-4", "system prompt", "user message")

    assert result is None


def test_ask_vision(mock_client, mock_vision_completion):
    mock_client.chat.completions.create.return_value = mock_vision_completion
    base64_image = base64.b64encode(b"test image data").decode()

    result = ask_vision(mock_client, "llama-3.2-11b-vision-preview", "system prompt", "user message", base64_image)
    assert result == "vision model response"


def test_get_combined_caption(mock_client, mock_completion):
    with patch("caption_train.llm.ask", return_value="combined caption") as mock_ask:
        result = get_combined_caption(
            mock_client, "gpt-4", "system prompt", "image caption", "pre-generated caption", max_tokens=100
        )

        mock_ask.assert_called_once()
        assert result == "combined caption"


def test_get_combined_caption_with_extra_args(mock_client, mock_completion):
    with patch("caption_train.llm.ask") as mock_ask:
        mock_ask.return_value = "combined caption"

        result = get_combined_caption(mock_client, "gpt-4", "system prompt", "image caption", "pre-generated caption")
        assert result is not None

        # Check if extra_body was passed with n_gpu_layers
        args = mock_ask.call_args.kwargs
        assert args.get("extra_body") == {"args": {"n_gpu_layers": 5}}


@pytest.mark.parametrize(
    "kwargs", [{"temperature": 0.7}, {"top_p": 0.9, "temperature": 0.5}, {"frequency_penalty": 1.0}]
)
def test_ask_with_kwargs(mock_client, mock_completion, kwargs):
    mock_client.chat.completions.create.return_value = mock_completion
    ask(mock_client, "model", "system", "message", **kwargs)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    for key, value in kwargs.items():
        assert call_kwargs[key] == value


def test_get_combined_caption_max_tokens():
    client = MagicMock()
    with patch("caption_train.llm.ask") as mock_ask:
        get_combined_caption(client, "model", "system", "image", "pregenerated", max_tokens=123)
        assert mock_ask.call_args.kwargs["max_tokens"] == 123


def test_ask_vision_correct_model():
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock()
    ask_vision(client, "wrong-model", "system", "message", "base64data")
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.2-11b-vision-preview"  # Model is hardcoded
