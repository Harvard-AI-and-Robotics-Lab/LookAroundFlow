import os
import csv
import pandas as pd
import shutil

def copy_images(input_folder, output_folder):
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for root, _, files in os.walk(input_folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(output_folder, file)

                # Avoid overwriting if same name exists
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(os.path.join(output_folder, f"{base}_{i}{ext}")):
                        i += 1
                    dst_path = os.path.join(output_folder, f"{base}_{i}{ext}")

                shutil.copy2(src_path, dst_path)
                count += 1

    print(f"Copied {count} image(s) from '{input_folder}' to '{output_folder}'.")


# === CONFIGURATION ===
clip_scores_csv = "clip_scores.csv"  # path to the clip score csv

source_folder = '/path/to/coco17/train'
destination_folder = '/path/to/coco17/train_bestcaption'

clip_scores_csv = os.path.join(source_folder, clip_scores_csv)

os.makedirs(destination_folder, exist_ok=True)

# === STEP 1: Read CLIP scores ===
clip_df = pd.read_csv(clip_scores_csv, dtype={"image_id": str})
# select the row with the maximum clip_score for each image_id
best_clip = clip_df.loc[clip_df.groupby("image_id")["clip_score"].idxmax()].reset_index(drop=True)

# === STEP 2: Load captions ===
# Expecting a folder where each image_id has a file with multiple captions (one per line)
# or one large CSV file where each row is (image_id, caption_index, caption)
captions = {}

for file in os.listdir(source_folder):
    if file.endswith(".txt"):
        image_id = os.path.splitext(file)[0]
        file_ = os.path.join(source_folder, file)
        with open(file_, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        captions[image_id] = {i: line for i, line in enumerate(lines)}

        out_path = os.path.join(destination_folder, f"{image_id}.txt")
        best_idx = max(captions[image_id].keys())
        best_caption = captions[image_id][best_idx]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(best_caption + "\n")

n_count = 0
# === STEP 3: Write best captions ===
for _, row in best_clip.iterrows():
    image_id = str(row["image_id"])
    best_idx = int(row["caption_index"])
    
    if image_id not in captions:
        print(f"[WARN] No captions found for {image_id}")
        continue
    
    if best_idx not in captions[image_id]:
        print(f"[WARN] Caption index {best_idx} not found for {image_id}")
        continue

    n_count += 1
    best_caption = captions[image_id][best_idx]
    out_path = os.path.join(destination_folder, f"{image_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(best_caption + "\n")

    # print(f"[INFO] Saved best caption for {image_id} (index {best_idx})")
print(f"✅ Total best captions saved: {n_count}")
copy_images(source_folder, destination_folder)

print("✅ All best captions saved to:", destination_folder)
