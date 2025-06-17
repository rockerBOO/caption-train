from flask import Flask, render_template, abort, send_from_directory, request, jsonify
import argparse
import tomli
from pathlib import Path

app = Flask(__name__)

# Default configuration
DEFAULT_CONFIG = {"images_dir": "images", "port": 5000, "host": "127.0.0.1", "debug": True, "template_dir": "templates"}


def load_config():
    """Load configuration from arguments or TOML file."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image gallery server")
    parser.add_argument("--images-dir", help="Directory containing images and caption files")
    parser.add_argument("--port", type=int, help="Port to run the server on")
    parser.add_argument("--host", help="Host to run the server on")
    parser.add_argument("--config", help="Path to TOML configuration file")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()

    # Load from TOML if specified
    if args.config:
        try:
            with open(args.config, "rb") as f:
                toml_config = tomli.load(f)
                config.update(toml_config)
        except Exception as e:
            print(f"Error loading config file: {e}")

    # Command-line args override TOML config
    if args.images_dir:
        config["images_dir"] = args.images_dir
    if args.port:
        config["port"] = args.port
    if args.host:
        config["host"] = args.host
    if args.debug:
        config["debug"] = args.debug

    return config


config = load_config()
app.config.update(config)


def get_images_and_captions():
    """Get all images and their captions from the configured directory."""
    images_dir = Path(app.config["images_dir"])
    image_files = []

    # Find all image files (supporting common formats)
    for ext in ["jpg", "jpeg", "png", "gif"]:
        image_files.extend(images_dir.glob(f"**/*.{ext}"))
        image_files.extend(images_dir.glob(f"**/*.{ext.upper()}"))

    images_and_captions = []

    for image_path in sorted(image_files):
        image_filename = image_path.name
        caption_path = image_path.with_suffix(".txt")

        caption = ""
        if caption_path.exists():
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()

        # Get relative path from the images directory
        rel_path = image_path.relative_to(images_dir)

        images_and_captions.append(
            {
                "id": len(images_and_captions),
                "filename": image_filename,
                "rel_path": str(rel_path),
                "full_path": str(image_path),
                "caption_path": str(caption_path),
                "caption": caption,
            }
        )

    return images_and_captions


@app.route("/")
def gallery():
    """Display the gallery page with all images and captions."""
    images = get_images_and_captions()
    return render_template("gallery.html", images=images)


@app.route("/image/<int:image_id>")
def image_detail(image_id):
    """Display a single image with its caption."""
    images = get_images_and_captions()

    if 0 <= image_id < len(images):
        image = images[image_id]
        prev_id = image_id - 1 if image_id > 0 else None
        next_id = image_id + 1 if image_id < len(images) - 1 else None
        return render_template("image.html", image=image, prev_id=prev_id, next_id=next_id)
    else:
        abort(404)


@app.route("/images/<path:path>")
def serve_image(path):
    """Serve images directly from the original directory."""
    return send_from_directory(app.config["images_dir"], path)


@app.route("/save_caption", methods=["POST"])
def save_caption():
    """Save the updated caption to the caption file."""
    try:
        data = request.get_json()
        image_id = int(data.get("image_id"))
        new_caption = data.get("caption", "")

        images = get_images_and_captions()
        if 0 <= image_id < len(images):
            image = images[image_id]
            print(image)
            print(data)
            caption_path = Path(image["caption_path"])

            # # Create parent directories if they don't exist
            # caption_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the new caption to the file
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(new_caption)

            return jsonify({"success": True, "message": "Caption saved successfully!"})
        else:
            return jsonify({"success": False, "message": "Image not found!"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


def create_templates():
    """Create the HTML templates if they don't exist."""
    template_dir = Path(app.config.get("template_dir", "templates"))
    template_dir.mkdir(exist_ok=True)

    gallery_template = template_dir / "gallery.html"
    if not gallery_template.exists():
        with open(gallery_template, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Image Gallery</title>
    <style>
        body {
            background-color: black;
        }
        h1 {
            text-align: center;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .image-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            transition: transform 0.3s;
        }
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        .image-card .caption {
            padding: 10px;
            height: 60px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        a {
            text-decoration: none;
            color: inherit;
        }
    </style>
</head>
<body>
    <h1>Image Gallery</h1>
    
    <div class="gallery">
        {% for image in images %}
        <a href="/image/{{ image.id }}">
            <div class="image-card">
                <img src="/images/{{ image.rel_path }}" alt="{{ image.filename }}">
                <div class="caption">{{ image.caption }}</div>
            </div>
        </a>
        {% endfor %}
    </div>
</body>
</html>""")

    image_template = template_dir / "image.html"
    if not image_template.exists():
        with open(image_template, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ image.filename }}</title>
    <style>
        body {
            background-color: black;
        }
        .image-container {
            text-align: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 80vh;
            border: 1px solid #ddd;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .caption {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f8f8;
            border-radius: 5px;
            line-height: 1.6;
        }
        .navigation {
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
        }
        .navigation a, .back-button {
            display: inline-block;
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }
        .navigation a:hover, .back-button:hover {
            background-color: #45a049;
        }
        .disabled {
            background-color: #cccccc;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="navigation">
        <a href="/" class="back-button">Back to Gallery</a>
        <div>
            {% if prev_id is not none %}
            <a href="/image/{{ prev_id }}">Previous</a>
            {% else %}
            <a class="disabled">Previous</a>
            {% endif %}
            
            {% if next_id is not none %}
            <a href="/image/{{ next_id }}">Next</a>
            {% else %}
            <a class="disabled">Next</a>
            {% endif %}
        </div>
    </div>
    
    <div class="image-container">
        <img src="/images/{{ image.rel_path }}" alt="{{ image.filename }}">
    </div>
    
    <div class="caption">
        <p>{{ image.caption }}</p>
        <p class="file-info">File: {{ image.rel_path }}</p>
    </div>
</body>
</html>""")


def create_sample_toml():
    """Create a sample TOML configuration file."""
    sample_config = """# Image Gallery Configuration

# Directory containing images and caption files
images_dir = "images"

# Server configuration
port = 5000
host = "127.0.0.1"
debug = true

# Template directory
template_dir = "templates"
"""

    with open("config.toml.sample", "w") as f:
        f.write(sample_config)


if __name__ == "__main__":
    # Create templates and sample config
    create_templates()
    create_sample_toml()

    # Get configuration
    host = app.config["host"]
    port = app.config["port"]
    debug = app.config["debug"]
    images_dir = app.config["images_dir"]

    print("Image Gallery Server")
    print("-------------------")
    print(f"Images directory: {images_dir}")
    print(f"Access the gallery at http://{host}:{port}/")

    app.run(host=host, port=port, debug=debug)
