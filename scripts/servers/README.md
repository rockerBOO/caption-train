# Server Scripts

This directory contains web servers and API endpoints for deploying trained models and providing web interfaces for caption generation and dataset management.

## Web Servers

### `server.py`

**Purpose**: General-purpose web API for caption generation

```bash
# Start the caption generation API server
uv run python scripts/servers/server.py \
    --model_path ./checkpoints \
    --host 0.0.0.0 \
    --port 8000

# With custom configuration
uv run python scripts/servers/server.py \
    --model_path ./trained_model \
    --host localhost \
    --port 5000 \
    --workers 4
```

**API Endpoints**:

- `POST /caption`: Generate caption for uploaded image
- `GET /health`: Health check endpoint
- `GET /model_info`: Model information and capabilities

**Example Usage**:

```bash
# Upload image for captioning
curl -X POST -F "image=@image.jpg" http://localhost:8000/caption

# Response
{
    "caption": "A red car parked on a city street",
    "confidence": 0.87,
    "processing_time": 0.3
}
```

### `florence-2-server.py`

**Purpose**: Specialized server for Florence-2 models with advanced features

```bash
# Start Florence-2 specific server
uv run python scripts/servers/florence-2-server.py \
    --model_path ./florence_checkpoints \
    --port 8001 \
    --enable_tasks

# With custom prompts and batch processing
uv run python scripts/servers/florence-2-server.py \
    --model_path ./florence_model \
    --port 8080 \
    --batch_size 8 \
    --max_workers 2
```

**Florence-2 Features**:

- Task-specific prompts (`<MORE_DETAILED_CAPTION>`, `<DETAILED_CAPTION>`)
- Batch image processing
- Multiple generation strategies
- Real-time streaming responses

**API Endpoints**:

- `POST /caption`: Basic captioning
- `POST /detailed_caption`: Detailed caption generation
- `POST /batch_caption`: Batch image processing
- `GET /tasks`: Available Florence-2 tasks

### `gallery.py`

**Purpose**: Web interface for browsing datasets and generated captions

```bash
# Start the gallery web interface
uv run python scripts/servers/gallery.py \
    --dataset_dir ./dataset \
    --captions_file ./captions.jsonl \
    --port 8080

# With custom styling and features
uv run python scripts/servers/gallery.py \
    --dataset_dir ./images \
    --captions_file ./results.jsonl \
    --port 3000 \
    --page_size 50 \
    --enable_search
```

**Gallery Features**:

- Browse images with generated captions
- Compare multiple caption sources
- Search and filter functionality
- Export selected results
- Side-by-side comparison views

## Deployment Configurations

### Development Setup

```bash
# Single-worker development server
uv run python scripts/servers/server.py \
    --model_path ./checkpoints \
    --host localhost \
    --port 8000 \
    --debug
```

### Production Setup

```bash
# Multi-worker production server
uv run python scripts/servers/server.py \
    --model_path ./production_model \
    --host 0.0.0.0 \
    --port 80 \
    --workers 8 \
    --timeout 30
```

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN uv sync --extra cu124
EXPOSE 8000
CMD ["uv", "run", "python", "scripts/servers/server.py", "--model_path", "./model", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t caption-server .
docker run -p 8000:8000 -v ./checkpoints:/app/model caption-server
```

## API Documentation

### Caption Generation API

#### POST /caption

Generate caption for a single image.

**Request**:

```bash
curl -X POST \
  -F "image=@photo.jpg" \
  -F "max_length=256" \
  -F "temperature=0.8" \
  http://localhost:8000/caption
```

**Response**:

```json
{
  "caption": "A sunset over mountain peaks with orange sky",
  "confidence": 0.89,
  "processing_time": 0.45,
  "model_info": {
    "model_type": "florence-2",
    "version": "1.0.0"
  }
}
```

#### POST /batch_caption

Process multiple images in a single request.

**Request**:

```bash
curl -X POST \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "batch_size=2" \
  http://localhost:8000/batch_caption
```

**Response**:

```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "caption": "A red car on a street",
      "confidence": 0.85
    },
    {
      "filename": "image2.jpg",
      "caption": "A mountain landscape",
      "confidence": 0.92
    }
  ],
  "total_processing_time": 0.8
}
```

### Health Check API

#### GET /health

Check server status and model availability.

**Response**:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "memory_usage": {
    "gpu_memory_used": "2.1GB",
    "gpu_memory_total": "24GB"
  },
  "uptime": "2h 45m"
}
```

## Configuration Options

### Server Configuration

```bash
--host 0.0.0.0          # Bind address (default: localhost)
--port 8000             # Port number (default: 8000)
--workers 4             # Number of worker processes (default: 1)
--timeout 30            # Request timeout in seconds (default: 30)
--max_request_size 10MB # Maximum upload size (default: 10MB)
```

### Model Configuration

```bash
--model_path ./model    # Path to trained model (required)
--batch_size 8          # Inference batch size (default: 1)
--device cuda           # Device to use (default: auto-detect)
--precision fp16        # Model precision (fp16/fp32, default: fp16)
```

### Generation Parameters

```bash
--max_length 256        # Maximum caption length (default: 256)
--temperature 0.8       # Sampling temperature (default: 1.0)
--top_p 0.9            # Top-p sampling (default: 1.0)
--num_beams 1          # Beam search beams (default: 1)
```

## Security Considerations

### Input Validation

- Image format validation
- File size limits
- Request rate limiting
- Input sanitization

### Authentication (Production)

```bash
# Add API key authentication
--api_key your_secret_key

# Or use environment variable
export API_KEY="your_secret_key"
```

**Authenticated Request**:

```bash
curl -X POST \
  -H "Authorization: Bearer your_secret_key" \
  -F "image=@photo.jpg" \
  http://localhost:8000/caption
```

### HTTPS Setup

```bash
# With SSL certificates
uv run python scripts/servers/server.py \
    --model_path ./model \
    --host 0.0.0.0 \
    --port 443 \
    --ssl_cert /path/to/cert.pem \
    --ssl_key /path/to/key.pem
```

## Monitoring and Logging

### Structured Logging

```bash
# Enable detailed logging
uv run python scripts/servers/server.py \
    --model_path ./model \
    --log_level INFO \
    --log_file server.log

# JSON formatted logs for production
--log_format json
```

### Metrics Collection

```bash
# Enable Prometheus metrics endpoint
--enable_metrics

# Metrics available at /metrics endpoint
curl http://localhost:8000/metrics
```

### Health Monitoring

```bash
# Health check for load balancers
curl http://localhost:8000/health

# Detailed system information
curl http://localhost:8000/system_info
```

## Performance Optimization

### GPU Utilization

```bash
# Optimize for GPU throughput
--batch_size 16         # Larger batches for better GPU utilization
--workers 1             # Single worker to avoid GPU memory conflicts
--precision fp16        # Faster inference with half precision
```

### Memory Management

```bash
# For limited GPU memory
--batch_size 1
--precision fp32
--offload_to_cpu       # Offload model parts to CPU when needed
```

### Load Balancing

```bash
# Multiple server instances
uv run python scripts/servers/server.py --port 8001 --model_path ./model &
uv run python scripts/servers/server.py --port 8002 --model_path ./model &

# Use nginx or similar for load balancing
```

## Client Integration Examples

### Python Client

```python
import requests

def generate_caption(image_path, server_url="http://localhost:8000"):
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{server_url}/caption",
            files={"image": f},
            data={"max_length": 256, "temperature": 0.8}
        )
    return response.json()

caption_data = generate_caption("photo.jpg")
print(caption_data["caption"])
```

### JavaScript Client

```javascript
async function generateCaption(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("max_length", "256");

  const response = await fetch("http://localhost:8000/caption", {
    method: "POST",
    body: formData,
  });

  return await response.json();
}
```

### cURL Examples

```bash
# Basic caption generation
curl -X POST -F "image=@photo.jpg" http://localhost:8000/caption

# With custom parameters
curl -X POST \
  -F "image=@photo.jpg" \
  -F "max_length=128" \
  -F "do_sample=true" \
  -F "temperature=0.7" \
  http://localhost:8000/caption

# Batch processing
curl -X POST \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg" \
  -F "images=@img3.jpg" \
  http://localhost:8000/batch_caption
```

## Troubleshooting

### Common Issues

**Server Won't Start**:

- Check port availability: `netstat -ln | grep :8000`
- Verify model path exists and is accessible
- Check GPU memory availability

**Slow Response Times**:

- Increase `--batch_size` for better throughput
- Use `--precision fp16` for faster inference
- Check GPU utilization with `nvidia-smi`

**Out of Memory Errors**:

- Reduce `--batch_size`
- Use `--precision fp16`
- Restart server to clear GPU memory

**Connection Refused**:

- Check firewall settings
- Verify host/port configuration
- Ensure server is running: `ps aux | grep server.py`

### Debug Mode

```bash
# Enable debug logging
uv run python scripts/servers/server.py \
    --model_path ./model \
    --debug \
    --log_level DEBUG
```

### Performance Profiling

```bash
# Enable profiling
uv run python scripts/servers/server.py \
    --model_path ./model \
    --enable_profiling \
    --profile_output ./profiles/
```

## Getting Help

```bash
uv run python scripts/servers/server.py --help
uv run python scripts/servers/florence-2-server.py --help
uv run python scripts/servers/gallery.py --help
```
