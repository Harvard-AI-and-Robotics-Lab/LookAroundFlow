import torch
from diffusers import StableDiffusion3Pipeline
import torch
from utils import AEuler
import json
from diffusers import StableDiffusion3Pipeline
import copy
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--method", choices=["aeuler", "euler"], default="aeuler")
argparser.add_argument("--num_inference_steps", type=int, default=5)
argparser.add_argument('--save_dir', type=str, default='./images')
argparser.add_argument('--prompt_dir', type=str, default='./captions_val2017.json')
args = argparser.parse_args()

device = args.device
save_dir = args.save_dir
num_inference_steps = args.num_inference_steps
method = args.method
prompt_dir = args.prompt_dir

pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16,local_files_only=True)
pipeline = pipeline.to(device)
if method == "aeuler":
    scheduler_old = copy.deepcopy(pipeline.scheduler)
    scheduler = AEuler.from_config(scheduler_old.config)
    scheduler._shift = 3.0
    pipeline.scheduler = scheduler


seed = 42
with open(prompt_dir, 'r') as f:
    data = json.load(f)

annotations = data['annotations']
image_id_to_caption = {}

for ann in annotations:
    image_id = ann['image_id']
    caption = ann['caption']
    if image_id not in image_id_to_caption:
        image_id_to_caption[image_id] = caption
    if len(image_id_to_caption) >= 5000:
        break
import os
pipeline.set_progress_bar_config(disable=True)
if args.method == "aeuler":
    path = os.path.join(save_dir, f'sd3_aeuler_{num_inference_steps}')
else:
    path = os.path.join(save_dir, f'sd3_euler_{num_inference_steps}')
os.makedirs(path, exist_ok=True)
import tqdm
for id,caption in tqdm.tqdm(image_id_to_caption.items()):
    generator = torch.Generator(device=device).manual_seed(id)
    images = pipeline(prompt=caption, num_inference_steps=num_inference_steps, guidance_scale=2, generator=generator).images[0]
    images.save(os.path.join(path,f"{id:012d}.png"))