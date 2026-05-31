import torch
from diffusers import StableDiffusion3Pipeline
from utils_sd35 import AFloPS_SD35
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--prompt", default="A cat is sitting on top of a bed")
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--output", default="afloops_sd35.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    
    # Load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    
    # Configure A-FloPS scheduler
    scheduler_config = dict(pipe.scheduler.config)
    scheduler_config.update({
        "lambda_min": -1.0,
        "lambda_max": 1.0,
    })
    pipe.scheduler = AFloPS_SD35.from_config(scheduler_config)
    
    # Generate
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    ).images[0]
    
    image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()