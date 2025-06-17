"""
Generic Caption Generation Server Script.

Run with:
uv run python generic_captioning_server.py --model_id=Salesforce/blip-image-captioning-large
"""

from caption_train.utils.server import load_config, setup_logging
from caption_train.servers import CaptionGenerationServer


def main():
    """
    Start the generic caption generation server.

    Supports command-line configuration for model selection and other parameters.
    """
    # Default configuration
    DEFAULT_CONFIG = {
        "model_id": "Salesforce/blip-image-captioning-large",
        "revision": None,
        "trust_remote_code": False,
        "peft_model": None,
        "task": None,
        "beams": 3,
        "save_captions": False,
        "caption_extension": ".txt",
        "max_token_length": 75,
        "host": "0.0.0.0",
        "port": 5123,
    }

    # Set up logging
    setup_logging()

    # Load configuration
    config = load_config(
        default_config=DEFAULT_CONFIG, description="Caption Generation Server", config_file_arg="--config"
    )

    # Create and run the server
    server = CaptionGenerationServer(
        model_id=config["model_id"],
        title="Generic Caption Generation Server",
        description="Caption generation with configurable models",
    )

    # Set additional server state
    if config.get("peft_model"):
        server.set_state("peft_model", config["peft_model"])

    # Set task if specified
    if config.get("task"):
        server.set_state("task", config["task"])

    # Run the server
    server.run(host=config["host"], port=config["port"])


if __name__ == "__main__":
    main()
