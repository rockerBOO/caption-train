"""
Image Gallery Server Script.

Run with:
uv run python image_gallery_server.py --images-dir /path/to/images
"""

from caption_train.utils.server import load_config, setup_logging
from caption_train.servers import ImageGalleryServer


def main():
    """
    Start the image gallery server.
    
    Supports command-line configuration for images directory and server settings.
    """
    # Default configuration
    DEFAULT_CONFIG = {
        "images_dir": "images",
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False,
        "template_dir": "templates"
    }
    
    # Set up logging
    setup_logging()
    
    # Load configuration
    config = load_config(
        default_config=DEFAULT_CONFIG,
        description="Image Gallery Server",
        config_file_arg="--config"
    )
    
    # Create and run the server
    server = ImageGalleryServer(
        images_dir=config['images_dir'], 
        title="Image Gallery Server",
        description="Web interface for browsing image galleries"
    )
    
    # Optionally set template directory
    if config.get('template_dir'):
        server.set_state('template_dir', config['template_dir'])
    
    # Run the server
    server.run(host=config['host'], port=config['port'])


if __name__ == "__main__":
    main()