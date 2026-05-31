#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Batch generate images from text prompt files")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Directory containing .txt prompt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated .png images")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for generated images (width and height)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    print("Loading Stable Diffusion 3 pipeline...")
    from models.stable_diffusion_3.pipeline_sd_3 import StableDiffusion3SGPAGPipeline
    
    pipe = StableDiffusion3SGPAGPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
    )
    device = "cuda"
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print("Pipeline loaded successfully.")
    
    # Find all prompt files
    prompt_files = sorted(Path(args.prompt_dir).glob("*.txt"))
    if not prompt_files:
        print(f"Error: No .txt files found in '{args.prompt_dir}'")
        return
    
    print(f"Found {len(prompt_files)} prompt files. Generating at {args.resolution}x{args.resolution} resolution.")
    
    # Process each prompt file
    for prompt_file in tqdm(prompt_files, desc="Generating images"):
        image_id = prompt_file.stem
        output_path = Path(args.output_dir) / f"{image_id}.png"
        
        if output_path.exists():
            continue
        
        try:
            prompt = prompt_file.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"\nError reading {prompt_file}: {e}")
            continue
        
        if not prompt:
            print(f"\nSkipping {image_id}: empty prompt")
            continue
        
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
        try:
            output = pipe(
                [prompt],
                width=args.resolution,
                height=args.resolution,
                num_inference_steps=25,
                guidance_scale=7,
                pag_scale=0.7,
                self_guidance_scale=3,
                self_guidance_shift_t=10,
                self_guidance_type="sg",
                generator=generator,
            )
            
            output.images[0].save(output_path)
            
        except Exception as e:
            print(f"\nError generating {image_id}: {e}")
            continue

if __name__ == "__main__":
    main()