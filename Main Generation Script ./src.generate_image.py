from diffusers import StableDiffusionPipeline
import torch
from model_loader import load_model
from utils import save_image, display_image
import argparse

def generate_image(prompt, model, steps=50, guidance=7.5, seed=None):
    """Generate image from text prompt"""
    if seed:
        torch.manual_seed(seed)
    
    with torch.inference_mode():
        image = model(
            prompt, 
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]
    
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="output.png", help="Output filename")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Load model (cached after first run)
    model = load_model()
    
    # Generate and save image
    image = generate_image(
        args.prompt,
        model,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed
    )
    
    save_image(image, f"outputs/{args.output}")
    display_image(image)
