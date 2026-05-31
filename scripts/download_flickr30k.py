import os
import argparse
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def download_and_process_dataset(dataset_name: str, output_dir: str):
    """
    Downloads an image-text dataset from Hugging Face and saves the images
    and all associated captions to a specified directory.

    Args:
        dataset_name (str): The name of the Hugging Face dataset.
        output_dir (str): The path to the folder where the data will be saved.
    """
    # Create the output directory if it doesn't exist
    print(f"📁 Ensuring output directory exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load the specified dataset from Hugging Face
    try:
        print(f"Downloading dataset '{dataset_name}'...")
        # The nlphuji/flickr_1k_test_image_text_retrieval dataset only has a 'test' split.
        dataset = load_dataset(dataset_name, split="test")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Please ensure the dataset name and split are correct.")
        return

    # Iterate over the dataset and save each image and its captions
    print(f"Processing {len(dataset)} samples...")
    for item in tqdm(dataset, desc="Saving images and captions"):
        image: Image.Image = item['image']
        
        # CORRECTED: 'caption' is a list of strings
        captions: list[str] = item['caption']
        
        filename: str = item['filename']
        
        # Extract the image_id (e.g., '1000092795') from the filename
        image_id = os.path.splitext(filename)[0]
        
        # Define the full paths for the image and text files
        image_path = os.path.join(output_dir, filename)
        caption_path = os.path.join(output_dir, f"{image_id}.txt")
        
        # Save the image, converting to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path)
        
        # CORRECTED: Save all captions, each on a new line
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(captions))
            
    print("\n✅ Download and processing complete.")
    print(f"All data from '{dataset_name}' has been saved to '{output_dir}'.")

def main():
    """Parses command-line arguments and runs the download function."""
    parser = argparse.ArgumentParser(
        description="Download an image-text dataset from Hugging Face and save images and captions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='datasets/flickr30k/train',
        help="The path to the folder where the dataset will be saved."
    )
    
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        # default="nlphuji/flickr_1k_test_image_text_retrieval",
        default="nlphuji/flickr30k",
        help="The name of the Hugging Face dataset to download."
    )
    
    args = parser.parse_args()
    
    download_and_process_dataset(args.dataset_name, args.output_dir)


if __name__ == '__main__':
    main()