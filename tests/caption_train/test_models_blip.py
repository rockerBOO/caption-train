import pytest
import torch
from unittest.mock import MagicMock, patch

from caption_train.models.blip import (
    setup_blip_model,
    setup_blip_vision2seq_model,
    create_blip_collate_fn,
    BLIP_TARGET_MODULES,
)
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
        model_id="Salesforce/blip-image-captioning-base",
        seed=42,
        log_with=None,
        name="test",
        prompt=None,
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
        rank=4,
        alpha=8,
        rslora=False,
        target_modules=BLIP_TARGET_MODULES,
        init_lora_weights="gaussian",
    )


def test_blip_target_modules():
    """Test that BLIP target modules are properly defined."""
    assert isinstance(BLIP_TARGET_MODULES, list)
    assert len(BLIP_TARGET_MODULES) > 0
    assert "q_proj" in BLIP_TARGET_MODULES
    assert "v_proj" in BLIP_TARGET_MODULES
    assert "k_proj" in BLIP_TARGET_MODULES


@patch("caption_train.models.blip.get_peft_model")
@patch("caption_train.models.blip.LoraConfig")
@patch("transformers.AutoProcessor.from_pretrained")
@patch("transformers.BlipForConditionalGeneration.from_pretrained")
def test_setup_blip_model_basic(
    mock_model_from_pretrained,
    mock_processor_from_pretrained,
    mock_lora_config,
    mock_get_peft_model,
    mock_training_config,
    mock_peft_config,
):
    """Test basic BLIP model setup."""
    # Setup mocks
    mock_model = MagicMock()
    mock_model.print_trainable_parameters = MagicMock()
    mock_model_from_pretrained.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor_from_pretrained.return_value = mock_processor

    # Create a proper LoRA config mock with required attributes
    mock_lora_config_instance = MagicMock()
    mock_lora_config_instance.task_type = "FEATURE_EXTRACTION"
    mock_lora_config_instance.base_model_name_or_path = "Salesforce/blip-image-captioning-base"
    mock_lora_config.return_value = mock_lora_config_instance

    mock_peft_model = MagicMock()
    mock_peft_model.print_trainable_parameters = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model

    # Call function
    model, processor = setup_blip_model("Salesforce/blip-image-captioning-base", mock_training_config, mock_peft_config)

    # Verify calls
    mock_model_from_pretrained.assert_called_once_with(
        "Salesforce/blip-image-captioning-base",
        quantization_config=None,
        torch_dtype=torch.float16,
    )

    mock_processor_from_pretrained.assert_called_once_with("Salesforce/blip-image-captioning-base")

    mock_lora_config.assert_called_once_with(
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        target_modules=BLIP_TARGET_MODULES,
        task_type="FEATURE_EXTRACTION",
        inference_mode=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )

    mock_get_peft_model.assert_called_once_with(mock_model, mock_lora_config_instance)
    mock_peft_model.print_trainable_parameters.assert_called_once()

    assert model == mock_peft_model
    assert processor == mock_processor


@patch("caption_train.models.blip.get_peft_model")
@patch("caption_train.models.blip.LoraConfig")
@patch("transformers.AutoProcessor.from_pretrained")
@patch("transformers.AutoModelForVision2Seq.from_pretrained")
def test_setup_blip_vision2seq_model_basic(
    mock_model_from_pretrained,
    mock_processor_from_pretrained,
    mock_lora_config,
    mock_get_peft_model,
    mock_training_config,
    mock_peft_config,
):
    """Test basic BLIP Vision2Seq model setup."""
    # Setup mocks
    mock_model = MagicMock()
    mock_model.print_trainable_parameters = MagicMock()
    mock_model_from_pretrained.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor_from_pretrained.return_value = mock_processor

    # Create a proper LoRA config mock with required attributes
    mock_lora_config_instance = MagicMock()
    mock_lora_config_instance.task_type = "FEATURE_EXTRACTION"
    mock_lora_config_instance.base_model_name_or_path = "Salesforce/blip-image-captioning-base"
    mock_lora_config.return_value = mock_lora_config_instance

    mock_peft_model = MagicMock()
    mock_peft_model.print_trainable_parameters = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model

    # Call function
    model, processor = setup_blip_vision2seq_model(
        "Salesforce/blip-image-captioning-base", mock_training_config, mock_peft_config
    )

    # Verify calls
    mock_model_from_pretrained.assert_called_once_with(
        "Salesforce/blip-image-captioning-base",
        quantization_config=None,
        torch_dtype=torch.float16,
    )

    mock_processor_from_pretrained.assert_called_once_with("Salesforce/blip-image-captioning-base")

    mock_lora_config.assert_called_once_with(
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        target_modules=BLIP_TARGET_MODULES,
        task_type="FEATURE_EXTRACTION",
        inference_mode=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )

    assert model == mock_peft_model
    assert processor == mock_processor


@patch("caption_train.models.blip.get_peft_model")
@patch("caption_train.models.blip.LoraConfig")
@patch("transformers.AutoProcessor.from_pretrained")
@patch("transformers.BlipForConditionalGeneration.from_pretrained")
def test_setup_blip_model_with_gradient_checkpointing(
    mock_model_from_pretrained,
    mock_processor_from_pretrained,
    mock_lora_config,
    mock_get_peft_model,
    mock_peft_config,
):
    """Test BLIP model setup with gradient checkpointing."""
    # Create training config with gradient checkpointing
    training_config = TrainingConfig(
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
        model_id="Salesforce/blip-image-captioning-base",
        seed=42,
        log_with=None,
        name="test",
        prompt=None,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        sample_every_n_epochs=None,
        sample_every_n_steps=None,
        save_every_n_epochs=None,
        save_every_n_steps=None,
    )

    # Setup mocks
    mock_model = MagicMock()
    mock_model.gradient_checkpointing_enable = MagicMock()
    mock_model_from_pretrained.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor_from_pretrained.return_value = mock_processor

    # Create a proper LoRA config mock with required attributes
    mock_lora_config_instance = MagicMock()
    mock_lora_config_instance.task_type = "FEATURE_EXTRACTION"
    mock_lora_config_instance.base_model_name_or_path = "Salesforce/blip-image-captioning-base"
    mock_lora_config.return_value = mock_lora_config_instance

    mock_peft_model = MagicMock()
    mock_peft_model.print_trainable_parameters = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model

    # Call function
    model, processor = setup_blip_model("Salesforce/blip-image-captioning-base", training_config, mock_peft_config)

    # Verify gradient checkpointing was enabled
    mock_model.gradient_checkpointing_enable.assert_called_once()


def test_create_blip_collate_fn():
    """Test creation of BLIP collate function."""
    # Create mock processor
    mock_processor = MagicMock()
    mock_processor.tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }

    # Create collate function
    collate_fn = create_blip_collate_fn(mock_processor, device="cpu")

    # Create sample batch
    batch = [
        {
            "pixel_values": torch.randn(3, 224, 224),
            "text": "A cat sitting on a table",
        },
        {
            "pixel_values": torch.randn(3, 224, 224),
            "text": "A dog running in a park",
        },
    ]

    # Call collate function
    result = collate_fn(batch)

    # Verify structure
    assert "pixel_values" in result
    assert "input_ids" in result
    assert "attention_mask" in result

    # Verify shapes
    assert result["pixel_values"].shape == (2, 3, 224, 224)

    # Verify processor was called
    mock_processor.tokenizer.assert_called_once_with(
        ["A cat sitting on a table", "A dog running in a park"],
        padding=True,
        return_tensors="pt",
    )


def test_create_blip_collate_fn_with_cuda():
    """Test BLIP collate function with CUDA device."""
    # Create mock processor
    mock_processor = MagicMock()
    mock_processor.tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    # Create collate function with CUDA device
    collate_fn = create_blip_collate_fn(mock_processor, device="cuda")

    # Create sample batch
    batch = [
        {
            "pixel_values": torch.randn(3, 224, 224),
            "text": "A test caption",
        }
    ]

    # Mock tensor.to method
    with patch.object(torch.Tensor, "to") as mock_to:
        mock_to.return_value = torch.tensor([[1, 2, 3]])
        result = collate_fn(batch)
        assert result is not None, "Collate function returned None"

        # Verify .to() was called with correct device
        assert mock_to.call_count >= 2  # Called for pixel_values, input_ids, attention_mask


@patch("caption_train.models.blip.get_peft_model")
@patch("caption_train.models.blip.LoraConfig")
@patch("transformers.AutoProcessor.from_pretrained")
@patch("transformers.BlipForConditionalGeneration.from_pretrained")
def test_setup_blip_model_custom_target_modules(
    mock_model_from_pretrained,
    mock_processor_from_pretrained,
    mock_lora_config,
    mock_get_peft_model,
    mock_training_config,
):
    """Test BLIP model setup with custom target modules."""
    # Create custom PEFT config
    custom_target_modules = ["query", "key", "value"]
    custom_peft_config = PeftConfig(
        rank=8,
        alpha=16,
        rslora=True,
        target_modules=custom_target_modules,
        init_lora_weights="pissa",
    )

    # Setup mocks
    mock_model = MagicMock()
    mock_model_from_pretrained.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor_from_pretrained.return_value = mock_processor

    # Create a proper LoRA config mock with required attributes
    mock_lora_config_instance = MagicMock()
    mock_lora_config_instance.task_type = "FEATURE_EXTRACTION"
    mock_lora_config_instance.base_model_name_or_path = "Salesforce/blip-image-captioning-base"
    mock_lora_config.return_value = mock_lora_config_instance

    mock_peft_model = MagicMock()
    mock_peft_model.print_trainable_parameters = MagicMock()
    mock_get_peft_model.return_value = mock_peft_model

    # Call function
    model, processor = setup_blip_model(
        "Salesforce/blip-image-captioning-base", mock_training_config, custom_peft_config
    )

    # Verify LoRA config used custom target modules
    mock_lora_config.assert_called_once_with(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=custom_target_modules,
        task_type="FEATURE_EXTRACTION",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="pissa",
    )
