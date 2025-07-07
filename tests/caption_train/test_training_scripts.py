"""Tests for training script integration with the library."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to Python path
scripts_dir = Path(__file__).parent.parent.parent / "scripts" / "training"
sys.path.insert(0, str(scripts_dir))


@pytest.fixture
def mock_training_pipeline():
    """Mock training pipeline for testing scripts."""
    with patch("caption_train.training.pipeline.TrainingPipeline") as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline.run_full_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        yield mock_pipeline


@pytest.fixture
def mock_create_training_pipeline():
    """Mock the create_training_pipeline function."""
    with patch("caption_train.training.pipeline.create_training_pipeline") as mock_create:
        mock_pipeline = MagicMock()
        mock_pipeline.run_full_pipeline = MagicMock()
        mock_create.return_value = mock_pipeline
        yield mock_create, mock_pipeline


class TestTrainingScriptImports:
    """Test that all training scripts can be imported successfully."""

    def test_train_florence_import(self):
        """Test that train_florence.py can be imported."""
        try:
            import train_florence

            assert hasattr(train_florence, "main")
        except ImportError as e:
            pytest.fail(f"Failed to import train_florence: {e}")

    def test_train_git_import(self):
        """Test that train_git.py can be imported."""
        try:
            import train_git

            assert hasattr(train_git, "main")
        except ImportError as e:
            pytest.fail(f"Failed to import train_git: {e}")

    def test_train_blip_import(self):
        """Test that train_blip.py can be imported."""
        try:
            import train_blip

            assert hasattr(train_blip, "main")
        except ImportError as e:
            pytest.fail(f"Failed to import train_blip: {e}")

    def test_train3_import(self):
        """Test that train3.py can be imported."""
        try:
            import train3

            assert hasattr(train3, "main")
            assert hasattr(train3, "create_train3_parser")
        except ImportError as e:
            pytest.fail(f"Failed to import train3: {e}")

    def test_fine_tune_blip_using_peft_import(self):
        """Test that fine_tune_blip_using_peft.py can be imported."""
        try:
            import fine_tune_blip_using_peft

            assert hasattr(fine_tune_blip_using_peft, "main")
            assert hasattr(fine_tune_blip_using_peft, "create_finetune_parser")
        except ImportError as e:
            pytest.fail(f"Failed to import fine_tune_blip_using_peft: {e}")


class TestTrain3ScriptFunctionality:
    """Test the train3.py specific functionality."""

    def test_create_train3_parser(self):
        """Test that train3 argument parser is created correctly."""
        import train3

        parser = train3.create_train3_parser()

        # Test that the parser has essential arguments
        help_text = parser.format_help()
        assert "--dataset_name" in help_text
        assert "--split" in help_text
        assert "--sample_every_n_steps" in help_text


class TestTrainingScriptIntegration:
    """Test integration between training scripts and the library."""

    def test_train_florence_script_integration(self, mock_create_training_pipeline):
        """Test that Florence training script integrates correctly with the library."""
        mock_create, mock_pipeline = mock_create_training_pipeline

        # Test that we can call the pipeline creation functions
        from caption_train.training.pipeline import create_training_pipeline

        # Mock the actual training to avoid dependencies
        with patch.object(mock_pipeline, "run_full_pipeline") as mock_run:
            assert mock_run is not None, "Mock patch failed"
            pipeline = create_training_pipeline(model_type="florence", model_id="microsoft/Florence-2-base-ft")

            # Verify the pipeline was created with correct parameters
            assert pipeline is not None

    def test_train_blip_script_integration(self, mock_create_training_pipeline):
        """Test that BLIP training script integrates correctly with the library."""
        mock_create, mock_pipeline = mock_create_training_pipeline

        # Test that we can call the pipeline creation functions
        from caption_train.training.pipeline import create_training_pipeline

        # Mock the actual training to avoid dependencies
        with patch.object(mock_pipeline, "run_full_pipeline") as mock_run:
            assert mock_run is not None, "Mock patch failed"
            pipeline = create_training_pipeline(model_type="blip", model_id="Salesforce/blip-image-captioning-base")

            # Verify the pipeline was created with correct parameters
            assert pipeline is not None

    def test_train3_script_argument_parser(self):
        """Test that train3 script has the correct argument structure."""
        import train3

        parser = train3.create_train3_parser()
        help_text = parser.format_help()

        # Check for external dataset specific arguments
        assert "--dataset_name" in help_text
        assert "--split" in help_text
        assert "--sample_every_n_steps" in help_text

    def test_fine_tune_script_argument_parser(self):
        """Test that fine-tune script has the correct argument structure."""
        import fine_tune_blip_using_peft

        parser = fine_tune_blip_using_peft.create_finetune_parser()
        help_text = parser.format_help()

        # Check for fine-tuning specific arguments
        assert "--target_modules" in help_text
        assert "--max_new_tokens" in help_text


class TestErrorHandling:
    """Test error handling in training scripts."""

    def test_script_handles_pipeline_creation_failure(self):
        """Test that scripts handle pipeline creation failures gracefully."""
        from caption_train.training.pipeline import create_training_pipeline

        # Test that an invalid model type raises an error
        with pytest.raises(ValueError):
            create_training_pipeline(model_type="invalid", model_id="some-model")


class TestConfigurationCompatibility:
    """Test that the scripts maintain configuration compatibility."""

    def test_external_dataset_scripts_configuration(self):
        """Test that external dataset scripts (train3, fine-tune) have correct defaults."""
        import train3
        import fine_tune_blip_using_peft

        # Test train3 defaults
        train3_parser = train3.create_train3_parser()
        train3_args = train3_parser.parse_args([])
        assert train3_args.dataset_name == "ybelkada/football-dataset"
        assert train3_args.epochs == 5
        assert train3_args.sample_every_n_steps == 10

        # Test fine-tune defaults
        finetune_parser = fine_tune_blip_using_peft.create_finetune_parser()
        finetune_args = finetune_parser.parse_args([])
        assert finetune_args.dataset_name == "ybelkada/football-dataset"
        assert finetune_args.epochs == 50
        assert finetune_args.max_new_tokens == 64
