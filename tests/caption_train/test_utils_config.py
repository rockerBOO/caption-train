import pytest
import tempfile
from argparse import Namespace
from pathlib import Path

try:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None
except ImportError:
    pass

from caption_train.utils.config import (
    load_toml_config,
    merge_args_with_config,
    validate_config_keys,
    get_config_section,
    update_args_from_config_section,
)


@pytest.fixture
def temp_toml_file():
    """Create a temporary TOML file for testing."""
    toml_content = """
[model]
name = "florence-2"
learning_rate = 0.001
batch_size = 4

[training]
epochs = 10
dropout = 0.1
seed = 42

[optimizer]
name = "adam"
weight_decay = 0.01

[dataset]
path = "/path/to/dataset"
num_workers = 4
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def invalid_toml_file():
    """Create an invalid TOML file for testing."""
    invalid_content = """
[model
name = "florence-2"
learning_rate = 0.001
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(invalid_content)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


def test_load_toml_config_success(temp_toml_file):
    """Test successful TOML config loading."""
    config = load_toml_config(temp_toml_file)

    assert isinstance(config, dict)
    assert "model" in config
    assert "training" in config
    assert "optimizer" in config
    assert "dataset" in config

    assert config["model"]["name"] == "florence-2"
    assert config["model"]["learning_rate"] == 0.001
    assert config["training"]["epochs"] == 10
    assert config["optimizer"]["name"] == "adam"


def test_load_toml_config_file_not_found():
    """Test loading non-existent TOML file."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_toml_config("nonexistent.toml")


def test_load_toml_config_invalid_toml(invalid_toml_file):
    """Test loading invalid TOML file."""
    with pytest.raises(ValueError, match="Invalid TOML configuration"):
        load_toml_config(invalid_toml_file)


def test_load_toml_config_with_path_object(temp_toml_file):
    """Test loading TOML with Path object."""
    config = load_toml_config(temp_toml_file)
    assert "model" in config


def test_merge_args_with_config_no_config():
    """Test merging args when no config file is provided."""
    args = Namespace(learning_rate=0.005, batch_size=8)
    result = merge_args_with_config(args, None)

    assert result.learning_rate == 0.005
    assert result.batch_size == 8


def test_merge_args_with_config_success(temp_toml_file):
    """Test successful merging of args with config."""
    args = Namespace(learning_rate=0.005, new_param="test")
    result = merge_args_with_config(args, temp_toml_file)

    # Args should take precedence
    assert result.learning_rate == 0.005

    # Config values should be added
    assert hasattr(result, "model")
    assert hasattr(result, "training")

    # New args should be preserved
    assert result.new_param == "test"


def test_validate_config_keys_success():
    """Test successful config validation."""
    config = {"model_name": "florence-2", "learning_rate": 0.001, "batch_size": 4}
    required_keys = ["model_name", "learning_rate"]

    # Should not raise any exception
    validate_config_keys(config, required_keys)


def test_validate_config_keys_missing():
    """Test config validation with missing keys."""
    config = {"model_name": "florence-2", "batch_size": 4}
    required_keys = ["model_name", "learning_rate", "epochs"]

    with pytest.raises(ValueError, match="Missing required configuration keys"):
        validate_config_keys(config, required_keys)


def test_get_config_section_exists():
    """Test getting existing config section."""
    config = {"model": {"name": "florence-2", "lr": 0.001}, "training": {"epochs": 10}}

    model_section = get_config_section(config, "model")
    assert model_section == {"name": "florence-2", "lr": 0.001}


def test_get_config_section_missing_with_default():
    """Test getting missing config section with default."""
    config = {"model": {"name": "florence-2"}}

    default = {"lr": 0.001, "batch_size": 4}
    section = get_config_section(config, "training", default)
    assert section == default


def test_get_config_section_missing_no_default():
    """Test getting missing config section without default."""
    config = {"model": {"name": "florence-2"}}

    section = get_config_section(config, "training")
    assert section == {}


def test_update_args_from_config_section():
    """Test updating args from config section."""
    config = {"training": {"learning_rate": 0.001, "batch_size": 4, "epochs": 10}}

    args = Namespace(learning_rate=0.005, new_param="test")

    result = update_args_from_config_section(args, config, "training")

    # Existing args should not be overwritten
    assert result.learning_rate == 0.005
    assert result.new_param == "test"

    # New values from config should be added
    assert result.batch_size == 4
    assert result.epochs == 10


def test_update_args_from_config_section_with_mapping():
    """Test updating args with key mapping."""
    config = {"training": {"lr": 0.001, "bs": 4, "epochs": 10}}

    args = Namespace(new_param="test")
    key_mapping = {"lr": "learning_rate", "bs": "batch_size"}

    result = update_args_from_config_section(args, config, "training", key_mapping)

    # Mapped keys should be correctly set
    assert result.learning_rate == 0.001
    assert result.batch_size == 4

    # Unmapped keys should use original names
    assert result.epochs == 10

    # Existing args should be preserved
    assert result.new_param == "test"


def test_update_args_from_config_section_missing_section():
    """Test updating args from missing config section."""
    config = {"model": {"name": "florence-2"}}
    args = Namespace(learning_rate=0.005)

    result = update_args_from_config_section(args, config, "training")

    # Args should remain unchanged
    assert result.learning_rate == 0.005

    # No new attributes should be added
    assert not hasattr(result, "epochs")


def test_update_args_from_config_section_none_values():
    """Test updating args when args have None values."""
    config = {"training": {"learning_rate": 0.001, "batch_size": 4}}

    args = Namespace(learning_rate=None, batch_size=8, epochs=None)

    result = update_args_from_config_section(args, config, "training")

    # None values should be overwritten by config
    assert result.learning_rate == 0.001

    # Non-None values should not be overwritten
    assert result.batch_size == 8

    # None values without config should remain None
    assert result.epochs is None
