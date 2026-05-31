import os
import os
os.environ["https_proxy"] = "http://127.0.0.1:37890"
os.environ["http_proxy"] = "http://127.0.0.1:37890"
os.environ["all_proxy"] = "socks5://127.0.0.1:37890"
import json
from PIL import Image
from tqdm import tqdm
import torch
import argparse
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import ImageReward as RM

# python eval_clip_ir.py --caption_json /coco_path/coco/annotations/captions_val2017.json --image_folder /generate_samples_path --save_csv

def load_data(caption_json_path, image_folder, max_items=5000):
    with open(caption_json_path, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    id_to_caption = {}
    for ann in annotations:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in id_to_caption:
            id_to_caption[image_id] = caption
        if len(id_to_caption) >= max_items:
            break
    return id_to_caption

def cal_clip_and_rm(prompt, image_path, clip_model, processor, rm_model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    clip_score = torch.nn.functional.cosine_similarity(
        outputs.image_embeds, outputs.text_embeds
    ).item()
    rm_score = rm_model.score(prompt, image)
    return clip_score, rm_score

def main(args):
    device = args.device
    caption_json = args.caption_json
    image_folder = args.image_folder
    max_items = args.max_items
    save_csv = args.save_csv

    print("Loading models...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    rm_model = RM.load("ImageReward-v1.0", device=device, download_root='cache_dir/ImageReward')

    print("Loading data...")
    id_to_caption = load_data(caption_json, image_folder, max_items)

    clip_scores = []
    rm_scores = []
    failed = []

    print("Evaluating...")
    for image_id, caption in tqdm(id_to_caption.items()):
        filename = os.path.join(image_folder, f"{image_id:012d}.png")
        if not os.path.exists(filename):
            failed.append(image_id)
            continue
        try:
            clip_score, rm_score = cal_clip_and_rm(caption, filename, clip_model, processor, rm_model, device)
            clip_scores.append(clip_score)
            rm_scores.append(rm_score)
        except Exception as e:
            print(f"Error on {image_id}: {e}")
            failed.append(image_id)

    print(f"\nEvaluated {len(clip_scores)} images (failed: {len(failed)}).")
    print(f"Mean CLIP score: {np.mean(clip_scores):.4f}")
    print(f"Mean ImageReward score: {np.mean(rm_scores):.4f}")

    if save_csv:
        import pandas as pd
        df = pd.DataFrame({
            "image_id": list(id_to_caption.keys())[:len(clip_scores)],
            "caption": list(id_to_caption.values())[:len(clip_scores)],
            "clip": clip_scores,
            "image_reward": rm_scores
        })
        csv_path = os.path.join(image_folder, "clip_ir_scores.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_json", type=str, required=True, help="Path to captions_val2017.json")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder with generated images")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA or CPU device")
    parser.add_argument("--max_items", type=int, default=5000, help="Max number of samples to evaluate")
    parser.add_argument("--save_csv", action="store_true", help="Save the results to CSV")

    args = parser.parse_args()
    main(args)
