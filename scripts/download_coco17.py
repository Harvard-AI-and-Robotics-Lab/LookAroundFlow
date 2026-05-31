import os
import json
import shutil
import requests
from datasets import load_dataset

# https://huggingface.co/datasets/phiyodr/coco2017

# Define the dataset name and the local directory to save the files
dataset_name = "phiyodr/coco2017"
local_dir = "datasets/coco17"

# Load the dataset from Hugging Face
dataset = load_dataset(dataset_name)

# Create the directory structure
os.makedirs(os.path.join(local_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(local_dir, 'validation'), exist_ok=True)
os.makedirs(os.path.join(local_dir, 'test'), exist_ok=True)

# Function to download and save files, ignoring TooManyRedirects errors and other exceptions
def download_and_save_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
    except requests.exceptions.TooManyRedirects:
        print(f"Too many redirects encountered for url: {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Process each split in the dataset
for split in dataset.keys():
    for idx, item in enumerate(dataset[split]):
        if idx % 1000 == 0:
            print(f"Processing {split} split: {idx}/{len(dataset[split])}")
        image_id = int(item['image_id'])
        image_url = item['coco_url']  # Assuming there's an 'image_url' field
        name = os.path.splitext(os.path.basename(item['file_name']))[0] # given file_name like train2017/000000522418.jpg, how to get 000000522418
        json_data = item  # Assuming the item itself is the JSON data

        # Define paths for image and JSON files
        image_path = os.path.join(local_dir, split, f"{name}.jpg")
        json_path = os.path.join(local_dir, split, f"{name}.json")

        # if the file image_path exists, continue the loop (skip the file)
        if os.path.exists(image_path):
            continue  

        # Download and save the image
        download_and_save_file(image_url, image_path)

        # Save the JSON data
        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file)

print("Dataset downloaded and organized successfully.")
