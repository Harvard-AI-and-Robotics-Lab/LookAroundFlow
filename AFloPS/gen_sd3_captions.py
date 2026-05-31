import torch
from diffusers import StableDiffusion3Pipeline
import copy
import argparse
import os
import tqdm
from utils import AEuler
from typing import Optional, Tuple

def parse_resolution(res_str: str) -> Tuple[int, int]:
    """Parse resolution string 'width,height' or 'size' into (width, height)"""
    parts = res_str.split(',')
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    elif len(parts) == 2:
        return int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Invalid resolution format: {res_str}. Use 'width,height' or a single integer.")

def validate_resolution(width: int, height: int):
    """Validate that resolution is compatible with SD3 (should be divisible by 16)"""
    if width % 16 != 0 or height % 16 != 0:
        print(f"Warning: Resolution ({width}x{height}) should be divisible by 16 for best results")

def set_seed(seed: Optional[int]):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--method", choices=["aeuler", "euler"], default="aeuler")
argparser.add_argument("--seed", type=int, default=42)
argparser.add_argument("--num_inference_steps", type=int, default=5)
argparser.add_argument("--guidance_scale", type=float, default=7)
argparser.add_argument('--output_dir', type=str, default='./images')
argparser.add_argument('--prompt_dir', type=str, default='./captions')
argparser.add_argument("--resolution", type=str, default="512", 
                       help="Image resolution as 'width,height' or single int for square (e.g., '512' or '1024,768')")
args = argparser.parse_args()

# Parse and validate resolution
try:
    width, height = parse_resolution(args.resolution)
    validate_resolution(width, height)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

set_seed(args.seed)

device = args.device
output_dir = args.output_dir
num_inference_steps = args.num_inference_steps
method = args.method
prompt_dir = args.prompt_dir

pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large", 
    torch_dtype=torch.bfloat16, 
    local_files_only=True
)
pipeline = pipeline.to(device)

if method == "aeuler":
    scheduler_old = copy.deepcopy(pipeline.scheduler)
    scheduler = AEuler.from_config(scheduler_old.config)
    scheduler._shift = 3.0
    pipeline.scheduler = scheduler

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Get all .txt files from prompt_dir
caption_files = [f for f in os.listdir(prompt_dir) if f.endswith('.txt')]

pipeline.set_progress_bar_config(disable=True)

# Process each caption file
for caption_file in tqdm.tqdm(caption_files):
    image_id_str = os.path.splitext(caption_file)[0]
    try:
        image_id = int(image_id_str)
    except ValueError:
        print(f"Skipping {caption_file}: invalid filename format")
        continue
    
    caption_path = os.path.join(prompt_dir, caption_file)
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
    except Exception as e:
        print(f"Error reading {caption_path}: {e}")
        continue
    
    if not caption:
        print(f"Skipping {caption_file}: empty caption")
        continue
    
    # Generate image with specified resolution
    generator = torch.Generator(device=device).manual_seed(image_id)
    image = pipeline(
        prompt=caption, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=args.guidance_scale, 
        generator=generator,
        height=height,
        width=width
    ).images[0]
    
    # Save as [image_id].png
    output_path = os.path.join(output_dir, f"{image_id_str}.png")
    image.save(output_path)