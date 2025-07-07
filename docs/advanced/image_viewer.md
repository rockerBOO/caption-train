# Image Viewer Server

## Overview

A lightweight HTTP server for browsing and viewing image datasets with captions.

## Features

- Serve images from a local directory
- Provide JSON listing of images with captions
- Support multiple image formats
- Simple web interface for dataset exploration

## Supported Image Formats

- PNG
- JPEG
- JPG
- WebP
- BMP

## Caption File Support

Looks for caption files with the following extensions:

- `.txt`
- `.caption`

## Usage

```bash
# Run image viewer server
uv run python viewer/server.py
```

### Configuration

- Serves images on `http://localhost:8123`
- Base URL path: `/img`
- Endpoint for image list: `/list.json`

## Endpoints

### `/list.json`

Returns a JSON array of images with their captions:

```json
[
  {
    "image": "image1.png",
    "caption": "A description of the image"
  },
  ...
]
```

## Workflow

1. Start the server with a specific directory
2. Point your browser to `http://localhost:8123`
3. Browse images and their corresponding captions

### Custom Directory Launch

```bash
# Specify a custom directory to serve
uv run python viewer/viewer.py /path/to/your/image/directory
```

## Use Cases

- Dataset preview
- Quick image and caption inspection
- Manual dataset validation
- Lightweight image gallery for machine learning datasets

## Customization

- Modify `viewer/viewer.py` to change default serving directory
- Adjust server configuration in `viewer/server.py` as needed

### Notes

- The default implementation includes a hardcoded path
- Always replace the path with your intended directory
- Ensure you have read permissions for the specified directory

## Performance Considerations

- Designed for local, small to medium-sized datasets
- Minimal server overhead
- No authentication or advanced features

## Best Practices

- Use for local development and dataset exploration
- Not recommended for production or large-scale deployments
- Ensure images and captions are in the same directory
