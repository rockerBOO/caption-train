# Captioning Servers

# Image Captioning Servers

## Generic Captioning Server

Provides a flexible web interface for image captioning models.

### Basic Usage

```bash
uv run python scripts/servers/generic_captioning_server.py \
    --model-id microsoft/Florence-2-base-ft \
    --host 0.0.0.0 \
    --port 8000
```

## Florence-2 Specific Captioning Server

Specialized server for Florence-2 model with advanced configuration.

### Basic Usage

```bash
uv run python scripts/servers/florence2_captioning_server.py \
    --base_model microsoft/Florence-2-base-ft \
    --task "<DETAILED_CAPTION>"
```

### Configuration Options

#### Model Selection

- `--base_model`: Florence-2 model variants
  - `microsoft/Florence-2-base-ft`
  - `microsoft/Florence-2-large-ft`
  - `microsoft/Florence-2-large`
  - `microsoft/Florence-2-base`
- `--peft_model`: Path to fine-tuned PEFT model

#### Captioning Options

- `--task`: Captioning detail level
  - `<CAPTION>`: Basic description
  - `<DETAILED_CAPTION>`: More elaborate description
  - `<MORE_DETAILED_CAPTION>`: Comprehensive analysis
- `--batch_size`: Images processed per batch
- `--max_token_length`: Maximum caption length

#### File Management

- `--save_captions`: Save captions alongside images
- `--overwrite`: Replace existing captions
- `--caption_extension`: File extension for captions

### Advanced Usage

```bash
uv run python scripts/servers/florence2_captioning_server.py \
    --base_model microsoft/Florence-2-large-ft \
    --peft_model ./checkpoints \
    --task "<MORE_DETAILED_CAPTION>" \
    --batch_size 4 \
    --max_token_length 256
```

## Performance Considerations

- Select appropriate model size
- Balance batch size with available GPU memory
- Adjust token length for detail vs. performance
- Use PEFT models for efficient fine-tuning

## Deployment Strategies

- Use behind reverse proxy
- Implement rate limiting
- Set up HTTPS
- Monitor resource usage
- Configure appropriate batch sizes
- Use GPU acceleration

## Scalability Tips

- Implement load balancing
- Use containerization (Docker)
- Set up horizontal scaling
- Monitor and auto-scale based on load
