import os
import torch
import argparse
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torchmetrics.functional.multimodal import clip_score

def compute_and_save_clip_scores(data_folder: str, output_path: str, model_name: str):
    """
    Computes CLIP scores for each image against each of its captions and saves to a CSV.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print(f"🖼️  Input folder: {os.path.abspath(data_folder)}")
    print(f"📝 Output CSV file: {output_path}")
    print(f"🧠 Model: {model_name}")

    try:
        image_files = [
            f for f in os.listdir(data_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{data_folder}' was not found.")
        return

    if not image_files:
        print(f"⚠️ No image files found in '{data_folder}'.")
        return

    with open(output_path, 'w') as f_out:
        # Write the CSV header
        f_out.write("image_id,caption_index,clip_score\n")

        for image_filename in tqdm(image_files, desc="Processing Images"):
            image_id = os.path.splitext(image_filename)[0]
            image_path = os.path.join(data_folder, image_filename)
            text_path = os.path.join(data_folder, f"{image_id}.txt")

            if not os.path.exists(text_path):
                continue

            try:
                # Load the image once per file
                image = Image.open(image_path).convert("RGB")
                img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor.to(device)

                # Read all captions from the corresponding text file
                with open(text_path, 'r', encoding='utf-8') as f_txt:
                    captions = [line.strip() for line in f_txt if line.strip()]

                if not captions:
                    continue

                # --- CORE CHANGE: Loop through each caption and score individually ---
                for i, caption in enumerate(captions):
                    # Calculate score for one image and one caption
                    score = clip_score(img_tensor, caption, model_name).detach()
                    # Write the result for this pair to the CSV file
                    f_out.write(f"{image_id},{i},{score.item():.4f}\n")

            except (UnidentifiedImageError, SyntaxError) as e:
                # Gracefully skip corrupted or unreadable images
                print(f"\n⚠️  Skipping corrupted image: {image_filename} ({e})")
                continue
            except Exception as e:
                print(f"\n❌ An unexpected error occurred with {image_filename}: {e}")
                continue

    print(f"\n✅ Success! CLIP scores have been saved to '{output_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute CLIP scores for image-caption pairs and save to a CSV file."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default='path/to/coco17/train',
        help="Path to the folder containing .jpg/.png and .txt files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="clip_scores.csv",
        help="Path to the output csv file. (Default: clip_scores.csv)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="Name of the CLIP model from Hugging Face. (Default: openai/clip-vit-base-patch16 | openai/clip-vit-large-patch14)"
    )
    args = parser.parse_args()
    
    # Construct the full path for the output file
    args.output_file = os.path.join(args.input_folder, args.output_file)

    compute_and_save_clip_scores(
        data_folder=args.input_folder,
        output_path=args.output_file,
        model_name=args.model_name
    )