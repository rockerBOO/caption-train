"""Configuration loading and management utilities."""

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from argparse import Namespace
from pathlib import Path
from typing import Any


def load_toml_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from TOML file.

    Args:
        config_path: Path to TOML configuration file

    Returns:
        Dictionary containing configuration values

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If TOML file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML configuration: {e}")


def merge_args_with_config(args: Namespace, config_path: str | Path | None = None) -> Namespace:
    """Merge command line arguments with TOML configuration file.

    Command line arguments take precedence over configuration file values.

    Args:
        args: Parsed command line arguments
        config_path: Optional path to TOML configuration file

    Returns:
        Namespace with merged configuration
    """
    if config_path is None:
        return args

    # Load configuration from file
    config_dict = load_toml_config(config_path)

    # Convert args to dict and merge with config
    args_dict = vars(args)
    merged_dict = {**config_dict, **args_dict}

    # Return as Namespace
    return Namespace(**merged_dict)


def validate_config_keys(config: dict[str, Any], required_keys: list[str]) -> None:
    """Validate that required configuration keys are present.

    Args:
        config: Configuration dictionary
        required_keys: List of required key names

    Raises:
        ValueError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def get_config_section(config: dict[str, Any], section: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get a specific section from configuration.

    Args:
        config: Full configuration dictionary
        section: Section name to extract
        default: Default values if section doesn't exist

    Returns:
        Configuration section dictionary
    """
    return config.get(section, default or {})


def update_args_from_config_section(
    args: Namespace, config: dict[str, Any], section: str, key_mapping: dict[str, str] | None = None
) -> Namespace:
    """Update arguments from a specific configuration section.

    Args:
        args: Arguments namespace to update
        config: Configuration dictionary
        section: Section name in config
        key_mapping: Optional mapping from config keys to arg names

    Returns:
        Updated arguments namespace
    """
    section_config = get_config_section(config, section)
    key_mapping = key_mapping or {}

    for config_key, value in section_config.items():
        # Map config key to argument name if mapping provided
        arg_key = key_mapping.get(config_key, config_key)

        # Only update if not already set in args
        if not hasattr(args, arg_key) or getattr(args, arg_key) is None:
            setattr(args, arg_key, value)

    return args
