"""Training pipeline orchestration utilities."""

from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoProcessor

from caption_train.datasets import Datasets, set_up_datasets, set_up_image_text_pair
from caption_train.models.florence import setup_florence_model, setup_git_model, FLORENCE_TARGET_MODULES
from caption_train.models.blip import setup_blip_model, BLIP_TARGET_MODULES
from caption_train.opt import get_accelerator, get_optimizer, get_scheduler
from caption_train.trainer import FileConfig, OptimizerConfig, PeftConfig, Trainer, TrainingConfig


class TrainingPipeline:
    """High-level training pipeline orchestrator.

    Handles the complete training setup including model loading, dataset preparation,
    optimizer configuration, and training execution.
    """

    def __init__(
        self,
        model_type: str = "florence",
        model_id: str = "microsoft/Florence-2-base-ft",
        target_modules: list[str] | None = None,
    ):
        """Initialize training pipeline.

        Args:
            model_type: Type of model to train ("florence", "git", etc.)
            model_id: HuggingFace model identifier
            target_modules: LoRA target modules (uses defaults if None)
        """
        self.model_type = model_type
        self.model_id = model_id
        self.target_modules = target_modules or self._get_default_target_modules()

        # Will be set during setup
        self.model = None
        self.processor = None
        self.accelerator = None
        self.optimizer = None
        self.scheduler = None
        self.datasets = None
        self.trainer = None

    def _get_default_target_modules(self) -> list[str]:
        """Get default target modules for the model type."""
        if self.model_type == "florence":
            return FLORENCE_TARGET_MODULES
        elif self.model_type == "git":
            return ["k_proj", "v_proj", "q_proj", "out_proj", "query", "key", "value"]
        elif self.model_type == "blip":
            return BLIP_TARGET_MODULES
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def setup_model(
        self,
        training_config: TrainingConfig,
        peft_config: PeftConfig,
    ) -> tuple[Any, AutoProcessor]:
        """Set up model and processor.

        Args:
            training_config: Training configuration
            peft_config: PEFT configuration

        Returns:
            Tuple of (model, processor)
        """
        if self.model_type == "florence":
            self.model, self.processor = setup_florence_model(self.model_id, training_config, peft_config)
        elif self.model_type == "git":
            self.model, self.processor = setup_git_model(self.model_id, training_config, peft_config)
        elif self.model_type == "blip":
            self.model, self.processor = setup_blip_model(self.model_id, training_config, peft_config)
        else:
            raise ValueError(f"Model type {self.model_type} not yet supported in pipeline")

        return self.model, self.processor

    def setup_accelerator(self, args: Any) -> Accelerator:
        """Set up accelerator for distributed training.

        Args:
            args: Command line arguments

        Returns:
            Configured accelerator
        """
        self.accelerator = get_accelerator(args)

        # Prepare model and processor with accelerator
        if self.model is not None and self.processor is not None:
            self.model, self.processor = self.accelerator.prepare(self.model, self.processor)

        return self.accelerator

    def setup_optimizer(
        self,
        training_config: TrainingConfig,
        optimizer_config: OptimizerConfig,
    ) -> Any:
        """Set up optimizer.

        Args:
            training_config: Training configuration
            optimizer_config: Optimizer configuration

        Returns:
            Configured optimizer
        """
        if self.model is None:
            raise ValueError("Model must be set up before optimizer")

        self.optimizer = get_optimizer(self.model, training_config.learning_rate, optimizer_config)

        if self.accelerator is not None:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        return self.optimizer

    def setup_scheduler(
        self,
        training_config: TrainingConfig,
        args: Any,
        steps_per_epoch: int,
    ) -> Any:
        """Set up learning rate scheduler.

        Args:
            training_config: Training configuration
            args: Command line arguments
            steps_per_epoch: Number of steps per epoch

        Returns:
            Configured scheduler or None
        """
        if not hasattr(args, "scheduler") or not args.scheduler:
            return None

        if self.optimizer is None:
            raise ValueError("Optimizer must be set up before scheduler")

        self.scheduler = get_scheduler(self.optimizer, training_config, args, steps_per_epoch=steps_per_epoch)

        if self.accelerator is not None:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        return self.scheduler

    def setup_datasets(
        self,
        training_config: TrainingConfig,
        dataset_config: FileConfig,
        dataset_path: Path,
    ) -> Datasets:
        """Set up datasets and dataloaders.

        Args:
            training_config: Training configuration
            dataset_config: Dataset configuration
            dataset_path: Path to dataset

        Returns:
            Configured datasets
        """
        if self.processor is None:
            raise ValueError("Processor must be set up before datasets")

        if dataset_path.is_dir():
            self.datasets = set_up_image_text_pair(
                self.model, self.processor, self.accelerator, training_config, dataset_config
            )
        else:
            self.datasets = set_up_datasets(dataset_path, self.processor, training_config, dataset_config)

        if self.accelerator is not None:
            self.datasets.accelerate(self.accelerator)

        return self.datasets

    def setup_trainer(
        self,
        training_config: TrainingConfig,
        file_config: FileConfig,
    ) -> Trainer:
        """Set up trainer.

        Args:
            training_config: Training configuration
            file_config: File configuration

        Returns:
            Configured trainer
        """
        if any(x is None for x in [self.model, self.processor, self.optimizer, self.accelerator, self.datasets]):
            raise ValueError("All components must be set up before trainer")

        self.trainer = Trainer(
            model=self.model,
            processor=self.processor,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            accelerator=self.accelerator,
            datasets=self.datasets,
            config=training_config,
            file_config=file_config,
        )

        return self.trainer

    def run_training(self) -> None:
        """Execute the training loop."""
        if self.trainer is None:
            raise ValueError("Trainer must be set up before running training")

        self.trainer.train()

    def run_full_pipeline(
        self,
        training_config: TrainingConfig,
        peft_config: PeftConfig,
        optimizer_config: OptimizerConfig,
        dataset_config: FileConfig,
        args: Any,
        dataset_path: Path,
    ) -> None:
        """Run the complete training pipeline.

        Args:
            training_config: Training configuration
            peft_config: PEFT configuration
            optimizer_config: Optimizer configuration
            dataset_config: Dataset configuration
            args: Command line arguments
            dataset_path: Path to dataset
        """
        # Set seed if provided
        if hasattr(args, "seed") and args.seed:
            set_seed(args.seed)

        # Setup all components
        print("Setting up model...")
        self.setup_model(training_config, peft_config)

        print("Setting up accelerator...")
        self.setup_accelerator(args)

        print("Setting up optimizer...")
        self.setup_optimizer(training_config, optimizer_config)

        print("Setting up datasets...")
        self.setup_datasets(training_config, dataset_config, dataset_path)

        print("Setting up scheduler...")
        self.setup_scheduler(training_config, args, len(self.datasets.train_dataloader))

        print("Setting up trainer...")
        self.setup_trainer(training_config, dataset_config)

        print("Starting training...")
        self.run_training()


def create_training_pipeline(
    model_type: str = "florence",
    model_id: str = "microsoft/Florence-2-base-ft",
) -> TrainingPipeline:
    """Create a training pipeline instance.

    Args:
        model_type: Type of model to train
        model_id: HuggingFace model identifier

    Returns:
        TrainingPipeline instance
    """
    return TrainingPipeline(model_type=model_type, model_id=model_id)
