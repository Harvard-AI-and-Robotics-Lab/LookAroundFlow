import os
import json
from datasets import load_dataset

# https://huggingface.co/datasets/cassiekang/cub200_dataset

# Define the dataset name and the local directory to save the files
dataset_name = "cassiekang/cub200_dataset"
local_dir = "datasets/cub_200"

# Load the dataset from Hugging Face
# This automatically downloads the data to your HF cache and decodes images to PIL objects
print(f"Loading dataset {dataset_name}...")
dataset = load_dataset(dataset_name)

# Process each split in the dataset (e.g., 'train', 'test')
for split in dataset.keys():
    # Create the directory structure for the current split
    split_dir = os.path.join(local_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Processing split: {split}")

    for idx, item in enumerate(dataset[split]):
        if idx % 100 == 0:
            print(f"Processing {split} split: {idx}/{len(dataset[split])}")
        
        # Generate a unique ID based on the split and index.
        image_id = f"{split}_{idx:05d}"
        
        # Extract the image object (PIL) and the text description
        image = item['image']
        text_content = item.get('text', '') # Safe get in case text is missing
        
        # Prepare metadata (everything except the PIL Image object)
        json_data = {k: v for k, v in item.items() if k != 'image'}
        json_data['img_id'] = image_id 

        # Define paths for image, text, and JSON files
        image_path = os.path.join(split_dir, f"{image_id}.jpg")
        text_path = os.path.join(split_dir, f"{image_id}.txt")
        json_path = os.path.join(split_dir, f"{image_id}.json")

        # if the file image_path exists, continue (skip)
        if os.path.exists(image_path):
            continue

        try:
            # Save the image directly using PIL's save method
            image.save(image_path)
            
            # Save the text description to a .txt file
            with open(text_path, 'w', encoding='utf-8') as text_file:
                text_file.write(str(text_content))

            # Save the JSON data
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, indent=4)
                
        except Exception as e:
            print(f"Error saving files for {image_id}: {e}")

print("Dataset downloaded and organized successfully.")