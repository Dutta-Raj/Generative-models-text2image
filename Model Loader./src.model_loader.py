from diffusers import StableDiffusionPipeline
import torch

def load_model(model_name="runwayml/stable-diffusion-v1-5", device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load Stable Diffusion model with safety checker disabled"""
    return StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        safety_checker=None
    ).to(device)
