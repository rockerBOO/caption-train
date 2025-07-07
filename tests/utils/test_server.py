"""Tests for server utility functions."""

import os
import tempfile
import pytest
from caption_train.utils.server import load_config, validate_path


def test_load_config_default():
    """Test loading default configuration."""
    default_config = {"port": 5000, "host": "localhost", "debug": False}

    # Mock sys.argv to simulate command-line arguments
    import sys

    orig_argv = sys.argv
    sys.argv = ["test_script.py"]

    try:
        config = load_config(default_config=default_config)
        assert config == default_config
    finally:
        sys.argv = orig_argv


def test_load_config_from_toml():
    """Test loading configuration from a TOML file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as temp_toml:
        toml_content = """
        port = 8080
        host = "0.0.0.0"
        debug = true
        """
        temp_toml.write(toml_content)
        temp_toml.close()

        # Mock sys.argv to simulate command-line arguments
        import sys

        orig_argv = sys.argv
        sys.argv = ["test_script.py", f"--config={temp_toml.name}"]

        try:
            config = load_config(default_config={"port": 5000, "host": "localhost", "debug": False})
            assert config == {"port": 8080, "host": "0.0.0.0", "debug": True}
        finally:
            sys.argv = orig_argv
            os.unlink(temp_toml.name)


def test_load_config_cli_override():
    """Test CLI arguments overriding TOML configuration."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as temp_toml:
        toml_content = """
        port = 8080
        host = "0.0.0.0"
        debug = true
        """
        temp_toml.write(toml_content)
        temp_toml.close()

        # Mock sys.argv to simulate command-line arguments
        import sys

        orig_argv = sys.argv
        sys.argv = ["test_script.py", f"--config={temp_toml.name}", "--port=9000"]

        try:
            config = load_config(default_config={"port": 5000, "host": "localhost", "debug": False})
            assert config == {"port": 9000, "host": "0.0.0.0", "debug": True}
        finally:
            sys.argv = orig_argv
            os.unlink(temp_toml.name)


def test_load_config_boolean_override():
    """Test boolean flag override."""
    import sys

    orig_argv = sys.argv
    sys.argv = ["test_script.py", "--debug"]

    try:
        config = load_config(default_config={"port": 5000, "host": "localhost", "debug": False})
        assert config == {"port": 5000, "host": "localhost", "debug": True}
    finally:
        sys.argv = orig_argv


def test_validate_path_existing():
    """Test validating an existing path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        validated_path = validate_path(temp_dir)
        assert validated_path.exists()
        assert validated_path.is_dir()


def test_validate_path_create():
    """Test creating a non-existent path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        new_dir = os.path.join(temp_dir, "new_subdir")
        validated_path = validate_path(new_dir, create_if_not_exists=True)
        assert validated_path.exists()
        assert validated_path.is_dir()


def test_validate_path_nonexistent_error():
    """Test that validating a non-existent path without create_if_not_exists raises an error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        non_existent_path = os.path.join(temp_dir, "non_existent_dir")

        with pytest.raises(ValueError, match="Path does not exist"):
            validate_path(non_existent_path)


def test_load_config_error_handling(capsys):
    """Test error handling when loading an invalid configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as temp_toml:
        # Invalid TOML content
        temp_toml.write("invalid toml content")
        temp_toml.close()

        # Mock sys.argv to simulate command-line arguments
        import sys

        orig_argv = sys.argv
        sys.argv = ["test_script.py", f"--config={temp_toml.name}"]

        try:
            config = load_config(default_config={"port": 5000})
            # Should fall back to default config
            assert config == {"port": 5000}

            # Check error was printed to stderr
            captured = capsys.readouterr()
            assert "Error loading config file" in captured.err
        finally:
            sys.argv = orig_argv
            os.unlink(temp_toml.name)
