"""Florence-2 model setup and configuration utilities."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor

from caption_train.trainer import TrainingConfig, PeftConfig


# Default target modules for Florence-2 LoRA fine-tuning
FLORENCE_TARGET_MODULES = [
    "qkv",
    "proj",
    "k_proj",
    "v_proj",
    "q_proj",
    "out_proj",
    "fc1",
    "fc2",
]


def setup_florence_model(
    model_id: str,
    training_config: TrainingConfig,
    peft_config: PeftConfig,
    revision: str = "refs/pr/1",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, AutoProcessor]:
    """Set up Florence-2 model with LoRA configuration.

    Args:
        model_id: HuggingFace model identifier for Florence-2
        training_config: Training configuration with quantization and checkpointing settings
        peft_config: PEFT configuration for LoRA fine-tuning
        revision: Model revision to use (default for Florence-2 PR)
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quantization_config,
        revision=revision,
        torch_dtype=torch_dtype,
    )

    # Setup gradient checkpointing if requested
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        print("Using gradient checkpointing")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Configure LoRA
    lora_config = LoraConfig(
        r=peft_config.rank,
        lora_alpha=peft_config.alpha,
        lora_dropout=training_config.dropout,
        bias="none",
        target_modules=peft_config.target_modules,
        task_type="CAUSAL_LM",
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


def setup_git_model(
    model_id: str,
    training_config: TrainingConfig,
    peft_config: PeftConfig,
    max_length: int = 2048,
) -> tuple[torch.nn.Module, AutoProcessor]:
    """Set up GIT model with LoRA configuration.

    Args:
        model_id: HuggingFace model identifier for GIT model
        training_config: Training configuration
        peft_config: PEFT configuration for LoRA fine-tuning
        max_length: Maximum sequence length

    Returns:
        Tuple of (model, processor) ready for training
    """
    from transformers import AutoConfig, AutoTokenizer

    # Setup quantization if requested
    quantization_config = None
    if training_config.quantize:
        from transformers import BitsAndBytesConfig

        if hasattr(training_config, "lora_bits"):
            if training_config.lora_bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            elif training_config.lora_bits == 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load config and disable features for LoRA
    config = AutoConfig.from_pretrained(model_id)
    config.gradient_checkpointing = False  # no gradient checkpointing for lora
    config.use_cache = False

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, config=config)

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Setup tokenizer padding
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Setting `pad_token` to `eos_token`: {tokenizer.eos_token}")

    # Set max length
    tokenizer.model_max_length = max_length
    print(f"Setting `block_size` to {max_length}")

    # Configure LoRA with GIT-specific target modules
    git_target_modules = [
        "k_proj",
        "v_proj",
        "q_proj",
        "out_proj",
        "query",
        "key",
        "value",
    ]

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=git_target_modules,
        inference_mode=False,
        r=peft_config.rank,
        lora_alpha=peft_config.alpha,
        lora_dropout=training_config.dropout,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config, adapter_name="CausalLM-LoRA")
    model.print_trainable_parameters()

    return model, processor
