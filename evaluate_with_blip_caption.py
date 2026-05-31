import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

def load_blip_model():
    """Load BLIP model and processor"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Let's use the original BLIP model which is more stable
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("Loading original BLIP model for more stable captioning...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        return model, processor, device, "BlipForConditionalGeneration"
    except Exception as e:
        print(f"Failed to load BLIP model: {str(e)}")
        raise e

def generate_caption(image_path, model, processor, device, model_type):
    """Generate a caption for the given image"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Use appropriate method based on model type
        if model_type == "BlipForConditionalGeneration":
            # Original BLIP model
            inputs = processor(image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=75)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
        else:
            # Fallback approach for other models
            inputs = processor(image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50)
            caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return "Error generating caption"

def read_ground_truth(caption_file):
    """Read ground truth caption from file"""
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {caption_file}: {e}")
        return ""

def calculate_metrics(generated_captions, ground_truth_captions):
    """Calculate BLEU, METEOR, and ROUGE scores"""
    # Prepare for BLEU calculation
    references = [[caption.split()] for caption in ground_truth_captions]
    hypotheses = [caption.split() for caption in generated_captions]
    
    # Calculate BLEU scores
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    # Calculate METEOR scores
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        meteor_scores.append(meteor_score(ref, hyp))
    meteor_avg = np.mean(meteor_scores)
    
    # Calculate ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores([' '.join(h) for h in hypotheses], 
                                   [' '.join(r[0]) for r in references], avg=True)
    
    metrics = {
        'BLEU-1': bleu1 * 100,
        'BLEU-2': bleu2 * 100,
        'BLEU-3': bleu3 * 100,
        'BLEU-4': bleu4 * 100,
        'METEOR': meteor_avg * 100,
        'ROUGE-1': rouge_scores['rouge-1']['f'] * 100,
        'ROUGE-2': rouge_scores['rouge-2']['f'] * 100,
        'ROUGE-L': rouge_scores['rouge-l']['f'] * 100
    }
    
    return metrics

def evaluate_captions(image_folder, caption_folder, output_file=None, model_name="Salesforce/blip-image-captioning-base"):
    """Evaluate generated captions against ground truth"""
    # Load BLIP model
    model, processor, device, model_type = load_blip_model()
    
    # Get image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images for evaluation")
    
    # Storage for results
    generated_captions = []
    ground_truth_captions = []
    results_dict = {}
    
    # Process each image
    for img_file in tqdm(image_files):
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_folder, img_file)
        caption_path = os.path.join(caption_folder, f"{img_id}.txt")
        
        # Check if ground truth caption exists
        if not os.path.exists(caption_path):
            print(f"Warning: No caption file found for {img_id}")
            continue
            
        # Generate caption and read ground truth
        generated = generate_caption(img_path, model, processor, device, model_type)
        ground_truth = read_ground_truth(caption_path)
        
        if generated and ground_truth:
            generated_captions.append(generated)
            ground_truth_captions.append(ground_truth)
            results_dict[img_id] = {
                'generated': generated,
                'ground_truth': ground_truth
            }
    
    # Calculate metrics
    if generated_captions and ground_truth_captions:
        metrics = calculate_metrics(generated_captions, ground_truth_captions)
        
        # Print results
        print("\nEvaluation Metrics:")
        for metric, score in metrics.items():
            print(f"{metric}: {score:.2f}")
            
        # Save results to file if specified
        if output_file:
            import json
            output = {
                'metrics': metrics,
                'captions': results_dict
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2)
                
        return metrics
    else:
        print("No valid caption pairs found for evaluation")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate BLIP v2 image captioning against ground truth')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--caption_folder', type=str, required=True, help='Path to the folder containing ground truth captions')
    parser.add_argument('--output_file', type=str, help='Path to save the evaluation results (optional)')
    parser.add_argument('--model_name', type=str, default="Salesforce/blip2-opt-2.7b", 
                        help='Model name or path (default: Salesforce/blip2-opt-2.7b)')
    
    args = parser.parse_args()
    
    # Install required packages if not already installed
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("NLTK resources loaded successfully")
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'nltk', 'rouge', 'transformers', 'pillow', 'torch', 'tqdm'])
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    
    # Print system information
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not installed, installing now...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'transformers'])
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    
    # Run evaluation
    print(f"\nStarting evaluation with images from: {args.image_folder}")
    print(f"Ground truth captions from: {args.caption_folder}")
    parent_dir = os.path.dirname(args.image_folder)
    args.output_file = f'{os.path.basename(args.image_folder)}_caption_results.json'
    args.output_file = os.path.join(parent_dir, args.output_file)
    print(f"Output file will be saved to: {args.output_file}")
    evaluate_captions(args.image_folder, args.caption_folder, args.output_file)