"""BLIP model setup and configuration utilities."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, BlipForConditionalGeneration

from caption_train.trainer import TrainingConfig, PeftConfig


# Default target modules for BLIP LoRA fine-tuning
BLIP_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "out_proj",
    "fc1",
    "fc2",
]


def setup_blip_model(
    model_id: str,
    training_config: TrainingConfig,
    peft_config: PeftConfig,
    torch_dtype: torch.dtype = torch.float16,
) -> tuple[torch.nn.Module, AutoProcessor]:
    """Set up BLIP model with LoRA configuration.

    Args:
        model_id: HuggingFace model identifier for BLIP
        training_config: Training configuration with quantization settings
        peft_config: PEFT configuration for LoRA fine-tuning
        torch_dtype: Model data type

    Returns:
        Tuple of (model, processor) ready for training
    """
    # Setup quantization if requested
    quantization_config = None
    if training_config.quantize:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    # Load base model
    if "blip" in model_id.lower():
        model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )
    else:
        # Use AutoModel for Vision2Seq for other models
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )

    # Setup gradient checkpointing if requested
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Using gradient checkpointing")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    # Configure LoRA
    lora_config = LoraConfig(
        r=peft_config.rank,
        lora_alpha=peft_config.alpha,
        lora_dropout=training_config.dropout,
        bias="none",
        target_modules=peft_config.target_modules,
        task_type="FEATURE_EXTRACTION",  # BLIP uses feature extraction task type
        inference_mode=False,
        use_rslora=peft_config.rslora,
        init_lora_weights=peft_config.init_lora_weights,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Count trainable modules for logging
    trainable_modules = sum(1 for name, param in model.named_parameters() if param.requires_grad)
    print(f"Trainable modules: {trainable_modules}")

    return model, processor


def setup_blip_vision2seq_model(
    model_id: str,
    training_config: TrainingConfig,
    peft_config: PeftConfig,
    torch_dtype: torch.dtype = torch.float16,
) -> tuple[torch.nn.Module, AutoProcessor]:
    """Set up BLIP Vision2Seq model with LoRA configuration.

    This is specifically for models that use AutoModelForVision2Seq.

    Args:
        model_id: HuggingFace model identifier
        training_config: Training configuration
        peft_config: PEFT configuration for LoRA fine-tuning
        torch_dtype: Model data type

    Returns:
        Tuple of (model, processor) ready for training
    """
    # Setup quantization if requested
    quantization_config = None
    if training_config.quantize:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    # Load base model using Vision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )

    # Setup gradient checkpointing if requested
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Using gradient checkpointing")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    # Configure LoRA with Vision2Seq specific settings
    lora_config = LoraConfig(
        r=peft_config.rank,
        lora_alpha=peft_config.alpha,
        lora_dropout=training_config.dropout,
        bias="none",
        target_modules=peft_config.target_modules,
        task_type="FEATURE_EXTRACTION",
        inference_mode=False,
        use_rslora=peft_config.rslora,
        init_lora_weights=peft_config.init_lora_weights,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Count trainable modules for logging
    trainable_modules = sum(1 for name, param in model.named_parameters() if param.requires_grad)
    print(f"Trainable modules: {trainable_modules}")

    return model, processor


def create_blip_collate_fn(processor: AutoProcessor, device: torch.device | None = None):
    """Create a collate function for BLIP models.

    Args:
        processor: AutoProcessor for tokenizing text
        device: Optional device to move tensors to

    Returns:
        Collate function for DataLoader
    """

    def collate_fn(batch):
        """Collate function for BLIP training."""
        processed_batch = {}

        for key in batch[0].keys():
            if key != "text":
                # Stack non-text tensors
                stacked = torch.stack([example[key] for example in batch])
                if device is not None:
                    stacked = stacked.to(device)
                processed_batch[key] = stacked
            else:
                # Process text with tokenizer
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch],
                    padding=True,
                    return_tensors="pt",
                )
                processed_batch["text"] = [example["text"] for example in batch]
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]

                if device is not None:
                    processed_batch["input_ids"] = processed_batch["input_ids"].to(device)
                    processed_batch["attention_mask"] = processed_batch["attention_mask"].to(device)

        return processed_batch

    return collate_fn
