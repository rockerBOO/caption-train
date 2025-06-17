"""Base server application classes for caption-train."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any


class CaptionServerApp:
    """
    Base FastAPI application for caption generation servers.
    
    Provides common functionality for managing server state and configuration.
    """
    
    def __init__(
        self, 
        title: str = "Caption Generation Server", 
        description: str = "API for image captioning",
        cors_origins: list[str] = ["*"]
    ):
        """
        Initialize the FastAPI application.
        
        Args:
            title: Name of the server application
            description: Description of the server's purpose
            cors_origins: List of allowed CORS origins
        """
        self.app = FastAPI(title=title, description=description)
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Server state
        self.state: Dict[str, Any] = {}
    
    def set_state(self, key: str, value: Any):
        """
        Set a state variable for the server.
        
        Args:
            key: Name of the state variable
            value: Value to store
        """
        self.state[key] = value
    
    def get_state(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a state variable.
        
        Args:
            key: Name of the state variable
            default: Value to return if key doesn't exist
        
        Returns:
            Value of the state variable or default
        """
        return self.state.get(key, default)
    
    def register_routes(self):
        """
        Method to be overridden by subclasses to register routes.
        Allows for a more modular approach to route definition.
        """
        raise NotImplementedError("Subclasses must implement route registration.")
    
    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """
        Run the FastAPI application.
        
        Args:
            host: Host to bind the server to
            port: Port to run the server on
        """
        import uvicorn
        
        # Ensure routes are registered before running
        self.register_routes()
        
        uvicorn.run(self.app, host=host, port=port)


class ImageGalleryServer(CaptionServerApp):
    """
    Specialized server for managing and serving an image gallery.
    """
    
    def __init__(
        self, 
        images_dir: Optional[str] = None, 
        title: str = "Image Gallery Server",
        description: str = "Web server for browsing image galleries"
    ):
        """
        Initialize the image gallery server.
        
        Args:
            images_dir: Directory containing images to serve
            title: Name of the server application
            description: Description of the server's purpose
        """
        super().__init__(title=title, description=description)
        
        # Set initial images directory if provided
        if images_dir:
            self.set_state("images_dir", images_dir)
    
    def register_routes(self):
        """
        Register routes for the image gallery server.
        """
        from fastapi import APIRouter, UploadFile, File
        from pathlib import Path
        
        router = APIRouter()
        
        @router.post("/set_images_dir")
        async def set_images_dir(images_dir: str):
            """Set the directory for serving images."""
            path = Path(images_dir).resolve()
            
            if not path.is_dir():
                raise ValueError(f"Invalid directory: {images_dir}")
            
            self.set_state("images_dir", str(path))
            
            # Get list of images
            images = sorted(path.glob("*.png"))
            
            return {
                "images_dir": str(path),
                "images": [str(img) for img in images]
            }
        
        @router.get("/images")
        async def get_images():
            """Retrieve list of images in the current directory."""
            images_dir = self.get_state("images_dir")
            
            if not images_dir:
                return {"images": []}
            
            path = Path(images_dir)
            images = sorted(path.glob("*.png"))
            
            return {"images": [str(img) for img in images]}
        
        # Add the router to the app
        self.app.include_router(router)


class CaptionGenerationServer(CaptionServerApp):
    """
    Specialized server for generating captions using VLM models.
    """
    
    def __init__(
        self, 
        model_id: str, 
        title: str = "Caption Generation Server",
        description: str = "AI-powered image captioning service"
    ):
        """
        Initialize the caption generation server.
        
        Args:
            model_id: Hugging Face model identifier
            title: Name of the server application
            description: Description of the server's purpose
        """
        super().__init__(title=title, description=description)
        
        # Set initial model configuration
        self.set_state("model_id", model_id)
    
    def register_routes(self):
        """
        Register routes for the caption generation server.
        """
        from fastapi import APIRouter, UploadFile, File
        from PIL import Image
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from peft import PeftModel
        from accelerate import Accelerator
        
        router = APIRouter()
        
        @router.post("/caption")
        @torch.no_grad()
        async def generate_caption(file: UploadFile = File(...)):
            """Generate a caption for an uploaded image."""
            # Load model configuration
            model_id = self.get_state("model_id")
            peft_model = self.get_state("peft_model")
            task = self.get_state("task", "<DETAILED_CAPTION>")
            
            # Model loading and inference logic
            accelerator = Accelerator()
            
            if peft_model:
                model = AutoModelForCausalLM.from_pretrained(model_id)
                model = PeftModel.from_pretrained(model, peft_model)
                processor = AutoProcessor.from_pretrained(model_id)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id)
                processor = AutoProcessor.from_pretrained(model_id)
            
            model, processor = accelerator.prepare(model, processor)
            model.eval()
            
            # Process image
            with Image.open(file.file) as image:
                batch = [image]
                inputs = processor(text=[task], images=batch, return_tensors="pt")
            
            # Generate caption
            with accelerator.autocast():
                generated_ids = model.generate(
                    **inputs.to(accelerator.device),
                    max_new_tokens=75,
                    num_beams=3,
                )
            
            generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            return {
                "image_file": file.filename,
                "caption": generated_captions[0]
            }
        
        # Add the router to the app
        self.app.include_router(router)