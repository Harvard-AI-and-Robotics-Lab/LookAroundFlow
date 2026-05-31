import json
import requests
import time
import argparse
from tqdm import tqdm
import os

def load_json_data(file_path):
    """Load caption data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_caption_pair(api_key, candidate_caption, reference_caption, model="gpt-3.5-turbo"):
    """
    Send a caption pair to ChatGPT API and get similarity score
    """
    prompt = f"""You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.
Candidate set:
{candidate_caption}
Reference set:
{reference_caption}
On a precise scale from 0 to 100, how likely is it that the candidate set is describing the same image as the reference set? (dict format, with a key "score", value between 0 and 100, and a key "reason" with a string value.)"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that evaluates caption similarity."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    print(f"Making API request with model: {model}")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        return {"score": 0, "reason": f"API error: {response.status_code}"}
    
    result = response.json()
    message_content = result["choices"][0]["message"]["content"]
    
    print(f"API response content: {message_content}")
    
    # Manually extract score from response
    try:
        # Try to parse JSON from the message
        score_dict = json.loads(message_content)
        return score_dict
    except json.JSONDecodeError:
        # If that fails, try to find the score in the text
        print("Could not parse response as JSON, attempting to extract score...")
        try:
            import re
            score_match = re.search(r'["\']score["\']\s*:\s*(\d+)', message_content)
            reason_match = re.search(r'["\']reason["\']\s*:\s*["\'](.*?)["\']', message_content)
            
            if score_match:
                score = int(score_match.group(1))
                reason = reason_match.group(1) if reason_match else "No reason provided"
                return {"score": score, "reason": reason}
            else:
                print("Could not extract score from response")
                return {"score": 0, "reason": "Failed to parse response"}
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {"score": 0, "reason": "Failed to parse response"}

def main():
    parser = argparse.ArgumentParser(description='Evaluate caption similarity using ChatGPT')
    parser.add_argument('--input', type=str, required=True, help='Path to the JSON file with caption pairs')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--output', type=str, default='similarity_results.json', help='Output file path')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini', help='OpenAI model to use, gpt-5-mini|gpt-4-turbo-preview|gpt-4o')
    parser.add_argument('--max-pairs', type=int, default=None, help='Maximum number of caption pairs to evaluate')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls in seconds')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    data = load_json_data(args.input)
    
    captions = data.get("captions", {})
    print(f"Found {len(captions)} caption pairs")
    
    results = {
        "caption_evaluations": {},
        "summary": {"total_score": 0, "count": 0}
    }
    
    # Get the list of caption pairs to process
    caption_ids = list(captions.keys())
    
    # Limit the number of pairs if specified
    if args.max_pairs is not None and args.max_pairs < len(caption_ids):
        caption_ids = caption_ids[:args.max_pairs]
        print(f"Limiting evaluation to {args.max_pairs} pairs as specified.")
    
    print(f"Processing {len(caption_ids)} caption pairs...")
    
    for image_id in tqdm(caption_ids):
        pair = captions[image_id]
        candidate = pair.get('generated', '')
        reference = pair.get('ground_truth', '')
        
        # Add a delay between API calls to avoid rate limiting
        time.sleep(args.delay)
        
        try:
            similarity = evaluate_caption_pair(args.api_key, candidate, reference, args.model)
            
            # Store result
            results["caption_evaluations"][image_id] = {
                "candidate": candidate,
                "reference": reference,
                "similarity": similarity
            }
            
            if "score" in similarity:
                score = similarity["score"]
                results["summary"]["total_score"] += score
                results["summary"]["count"] += 1
                print(f"Image {image_id}: Score = {score}/100")
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            # Continue with the next pair
    
    # Calculate average score before saving
    if results["summary"]["count"] > 0:
        avg_score = results["summary"]["total_score"] / results["summary"]["count"]
        results["summary"]["average_score"] = avg_score

    parent_dir = os.path.dirname(args.input)
    args.output = os.path.splitext(os.path.basename(args.input))[0]
    args.output = f'{os.path.basename(args.output)}_clair.json'
    args.output = os.path.join(parent_dir, args.output)
    # Save the test result
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {args.output}")
    if results["summary"]["count"] > 0:
        # avg_score = results["summary"]["total_score"] / results["summary"]["count"]
        # results["summary"]["average_score"] = avg_score
        print(f"Average CLAIR similarity score: {avg_score:.2f}")
        print(f"Evaluated {results['summary']['count']} out of {len(captions)} caption pairs successfully.")

if __name__ == "__main__":
    main()