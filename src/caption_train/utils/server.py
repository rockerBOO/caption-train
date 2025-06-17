"""Shared utilities for server configuration and management."""

import argparse
import tomli
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(
    default_config: Optional[Dict[str, Any]] = None,
    config_file_arg: str = "--config",
    description: str = "Server configuration loader",
) -> Dict[str, Any]:
    """
    Load configuration from command-line arguments and optional TOML file.

    Args:
        default_config: Default configuration dictionary
        config_file_arg: Name of the configuration file argument
        description: Description for argument parser

    Returns:
        Merged configuration dictionary
    """
    default_config = default_config or {}

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(config_file_arg, help="Path to TOML configuration file")

    # Add default config keys as arguments with type inference
    for key, value in default_config.items():
        arg_name = f"--{key.replace('_', '-')}"
        arg_type = type(value)
        if arg_type is bool:
            # For boolean flags, we want to set to True if flag is present
            parser.add_argument(arg_name, action="store_true", help=f"Override {key} from configuration")
        else:
            parser.add_argument(arg_name, type=arg_type, help=f"Override {key} from configuration")

    # Use parse_known_args to ignore unexpected arguments
    args, unknown = parser.parse_known_args()
    config = default_config.copy()

    # Load from TOML if specified
    config_path = getattr(args, config_file_arg.lstrip("-"))
    if config_path:
        try:
            with open(config_path, "rb") as f:
                toml_config = tomli.load(f)
                config.update(toml_config)
        except Exception as e:
            print(f"Error loading config file: {e}", file=sys.stderr)

    # Override config with command-line arguments
    for key, default_value in default_config.items():
        cli_arg = getattr(args, key.replace("_", "-"), None)
        # Handle boolean flags differently
        if isinstance(default_value, bool):
            # For boolean flags, only change if the flag is present
            if getattr(args, key.replace("_", "-")) is True:
                config[key] = True
        elif cli_arg is not None:
            # For non-boolean types, convert to correct type
            config[key] = type(default_value)(cli_arg)

    return config


def validate_path(path_str: str, create_if_not_exists: bool = False) -> Path:
    """
    Validate and potentially create a path.

    Args:
        path_str: Path as a string
        create_if_not_exists: Create directory if it doesn't exist

    Returns:
        Validated Path object
    """
    path = Path(path_str).expanduser().resolve()

    if create_if_not_exists:
        path.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    return path


def setup_logging(log_level: str = "INFO"):
    """
    Set up basic logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
