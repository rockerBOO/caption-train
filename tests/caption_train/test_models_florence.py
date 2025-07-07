import pytest
import torch
from unittest.mock import MagicMock, patch

from caption_train.models.florence import setup_florence_model, setup_git_model, FLORENCE_TARGET_MODULES
from caption_train.trainer import TrainingConfig, PeftConfig


@pytest.fixture
def mock_training_config():
    return TrainingConfig(
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=2,
        epochs=1,
        max_length=512,
        shuffle_captions=False,
        frozen_parts=0,
        caption_dropout=0.0,
        quantize=False,
        device="cpu",
        model_id="microsoft/Florence-2-base-ft",
        seed=42,
        log_with="wandb",
        name="test",
        prompt="<MORE_DETAILED_CAPTION>",
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        sample_every_n_epochs=None,
        sample_every_n_steps=None,
        save_every_n_epochs=None,
        save_every_n_steps=None,
    )


@pytest.fixture
def mock_peft_config():
    return PeftConfig(
        rank=4, alpha=8, rslora=False, target_modules=FLORENCE_TARGET_MODULES, init_lora_weights="gaussian"
    )


@pytest.fixture
def mock_quantized_training_config():
    return TrainingConfig(
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=2,
        epochs=1,
        max_length=512,
        shuffle_captions=False,
        frozen_parts=0,
        caption_dropout=0.0,
        quantize=True,
        device="cpu",
        model_id="microsoft/Florence-2-base-ft",
        seed=42,
        log_with="wandb",
        name="test",
        prompt="<MORE_DETAILED_CAPTION>",
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        sample_every_n_epochs=None,
        sample_every_n_steps=None,
        save_every_n_epochs=None,
        save_every_n_steps=None,
    )


def test_florence_target_modules():
    """Test that Florence target modules are properly defined."""
    assert isinstance(FLORENCE_TARGET_MODULES, list)
    assert len(FLORENCE_TARGET_MODULES) > 0
    assert "qkv" in FLORENCE_TARGET_MODULES
    assert "proj" in FLORENCE_TARGET_MODULES


@patch("caption_train.models.florence.get_peft_model")
@patch("caption_train.models.florence.LoraConfig")
@patch("transformers.AutoProcessor.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
def test_setup_florence_model_basic(
    mock_model_from_pretrained,
    mock_processor_from_pretrained,
    mock_lora_config,
    mock_get_peft_model,
    mock_training_config,
    mock_peft_config,
):
    """Test basic Florence model setup without quantization."""
    # Setup mocks
    mock_model = MagicMock()
    mock_model.print_trainable_parameters = MagicMock()
    mock_model_from_pretrained.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor_from_pretrained.return_value = mock_processor

    # Create a proper LoRA config mock with required attributes
    mock_lora_config_instance = MagicMock()
    mock_lora_config.return_value = mock_lora_config_instance

    mock_peft_model = MagicMock()
    mock_peft_model.named_parameters.return_value = [
        ("param1", MagicMock(requires_grad=True)),
        ("param2", MagicMock(requires_grad=False)),
        ("param3", MagicMock(requires_grad=True)),
    ]
    mock_peft_model.print_trainable_parameters = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model

    # Call function
    model, processor = setup_florence_model("microsoft/Florence-2-base-ft", mock_training_config, mock_peft_config)

    # Verify calls
    mock_model_from_pretrained.assert_called_once_with(
        "microsoft/Florence-2-base-ft",
        trust_remote_code=True,
        quantization_config=None,
        revision="refs/pr/1",
        torch_dtype=torch.bfloat16,
    )

    mock_processor_from_pretrained.assert_called_once_with("microsoft/Florence-2-base-ft", trust_remote_code=True)

    mock_get_peft_model.assert_called_once()
    mock_peft_model.print_trainable_parameters.assert_called_once()

    assert model == mock_peft_model
    assert processor == mock_processor


@patch("caption_train.models.florence.get_peft_model")
@patch("caption_train.models.florence.LoraConfig")
@patch("transformers.BitsAndBytesConfig")
@patch("transformers.AutoProcessor.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
def test_setup_florence_model_quantized(
    mock_model_from_pretrained,
    mock_processor_from_pretrained,
    mock_bnb_config,
    mock_lora_config,
    mock_get_peft_model,
    mock_quantized_training_config,
    mock_peft_config,
):
    """Test Florence model setup with quantization and gradient checkpointing."""
    # Setup mocks
    mock_quantization_config = MagicMock()
    mock_bnb_config.return_value = mock_quantization_config

    # Create a proper LoRA config mock with required attributes
    mock_lora_config_instance = MagicMock()
    mock_lora_config_instance.task_type = "CAUSAL_LM"
    mock_lora_config_instance.base_model_name_or_path = "microsoft/Florence-2-base-ft"
    mock_lora_config.return_value = mock_lora_config_instance

    mock_model = MagicMock()
    mock_model.gradient_checkpointing_enable = MagicMock()
    mock_model_from_pretrained.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor_from_pretrained.return_value = mock_processor

    mock_peft_model = MagicMock()
    mock_peft_model.named_parameters.return_value = [
        ("param1", MagicMock(requires_grad=True)),
    ]
    mock_peft_model.print_trainable_parameters = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model

    # Call function
    model, processor = setup_florence_model(
        "microsoft/Florence-2-base-ft", mock_quantized_training_config, mock_peft_config
    )

    # Verify quantization config was created
    mock_bnb_config.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Verify model was loaded with quantization
    mock_model_from_pretrained.assert_called_once_with(
        "microsoft/Florence-2-base-ft",
        trust_remote_code=True,
        quantization_config=mock_quantization_config,
        revision="refs/pr/1",
        torch_dtype=torch.bfloat16,
    )

    # Verify gradient checkpointing was enabled
    mock_model.gradient_checkpointing_enable.assert_called_once()


@patch("caption_train.models.florence.get_peft_model")
@patch("caption_train.models.florence.LoraConfig")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoConfig.from_pretrained")
@patch("transformers.AutoProcessor.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
def test_setup_git_model_basic(
    mock_model_from_pretrained,
    mock_processor_from_pretrained,
    mock_config_from_pretrained,
    mock_tokenizer_from_pretrained,
    mock_lora_config,
    mock_get_peft_model,
    mock_training_config,
    mock_peft_config,
):
    """Test basic GIT model setup."""
    # Setup mocks
    mock_config = MagicMock()
    mock_config_from_pretrained.return_value = mock_config

    # Create a proper LoRA config mock with required attributes
    mock_lora_config_instance = MagicMock()
    mock_lora_config_instance.task_type = "CAUSAL_LM"
    mock_lora_config_instance.base_model_name_or_path = "microsoft/Florence-2-base-ft"
    mock_lora_config.return_value = mock_lora_config_instance

    mock_model = MagicMock()
    mock_model_from_pretrained.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor_from_pretrained.return_value = mock_processor

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.pad_token_id = None
    mock_tokenizer.eos_token = "</s>"
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    mock_peft_model = MagicMock()
    mock_peft_model.print_trainable_parameters = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model

    # Call function
    model, processor = setup_git_model("microsoft/git-base", mock_training_config, mock_peft_config)

    # Verify config modifications
    assert mock_config.gradient_checkpointing is False
    assert mock_config.use_cache is False

    # Verify tokenizer padding setup
    assert mock_tokenizer.pad_token == "</s>"
    assert mock_tokenizer.pad_token_id == 2
    assert mock_tokenizer.model_max_length == 2048

    # Verify model components
    assert model == mock_peft_model
    assert processor == mock_processor


def test_setup_git_model_with_existing_pad_token(mock_training_config, mock_peft_config):
    """Test GIT model setup when tokenizer already has pad token."""
    with (
        patch("caption_train.models.florence.get_peft_model") as mock_get_peft_model,
        patch("caption_train.models.florence.LoraConfig") as mock_lora_config,
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_from_pretrained,
        patch("transformers.AutoProcessor.from_pretrained") as mock_processor_from_pretrained,
        patch("transformers.AutoConfig.from_pretrained") as mock_config_from_pretrained,
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_from_pretrained,
    ):
        # Setup mocks
        mock_config = MagicMock()
        mock_config_from_pretrained.return_value = mock_config

        mock_lora_config_instance = MagicMock()
        mock_lora_config.return_value = mock_lora_config_instance

        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model

        mock_processor = MagicMock()
        mock_processor_from_pretrained.return_value = mock_processor

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_peft_model = MagicMock()
        mock_peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft_model.return_value = mock_peft_model

        # Call function
        model, processor = setup_git_model(
            "microsoft/git-base", mock_training_config, mock_peft_config, max_length=1024
        )

        # Verify tokenizer wasn't modified since it already had pad token
        assert mock_tokenizer.pad_token == "<pad>"
        assert mock_tokenizer.pad_token_id == 1
        assert mock_tokenizer.model_max_length == 1024
