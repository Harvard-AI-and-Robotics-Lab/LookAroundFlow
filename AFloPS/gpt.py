from openai import OpenAI
import base64
import io
import json
import concurrent.futures
from tqdm import tqdm
from PIL import Image
from functools import partial

# 初始化客户端
client = OpenAI(
    api_key='APIKEY',
    base_url='https://api.gpt.ge/v1'
)

def encode_image(image_path, size=(256, 256)):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def compare_images(image1_path, image2_path, prompt, model, temperature):
    try:
        image_base64_1 = encode_image(image1_path)
        image_base64_2 = encode_image(image2_path)
        
        messages = [{
            "role": "system",
            "content": """You are an image quality assessment assistant. You will receive two images with a prompt for comparison.
Task: Evaluate both images considering two aspects:
Image quality and authenticity
How well each image aligns with the given prompt

Rules:
- Return 0 if the first image is better overall
- Return 1 if the second image is better overall
- Must return ONLY 0 or 1
- Do not provide any explanation or additional text
- Do not include any punctuation marks"""
        }, {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Based on this prompt: '{prompt}', compare these two images. Consider both image quality/authenticity and prompt alignment. Return 0 if the first image is better overall, return 1 if the second is better. Return only the number 0 or 1."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64_1}"}},
                {"type": "text", "text": "Here is the second image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64_2}"}},
                {"type": "text", "text": "Return only 0 or 1"}
            ]
        }]
        
        response = client.chat.completions.create(
            model=model, 
            messages=messages, 
            temperature=temperature
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_image_filename(image_id, extension='.png'):
    return f"{image_id:012d}{extension}"

def load_coco_data(coco_file):
    with open(coco_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    id_to_image_id = {}
    id_to_prompt = {}
    print(f'total annotations: {len(annotations)}')
    
    for ann in annotations:
        ann_id = ann['id']
        image_id = ann['image_id'] 
        caption = ann['caption']
        
        id_to_image_id[ann_id] = image_id
        id_to_prompt[ann_id] = caption
    
    return id_to_image_id, id_to_prompt

def process_single_comparison(id, base_path1, base_path2, id_to_image_id, id_to_prompt, image_extension, model, temperature):
    try:
        # 获取实际的image_id和prompt
        image_id = id_to_image_id.get(id)
        prompt = id_to_prompt.get(id)
        
        if image_id is None or prompt is None:
            print(f"No mapping found for id {id}")
            return id, None
            
        filename = get_image_filename(image_id, image_extension)
        image1 = f'{base_path1}/{filename}'
        image2 = f'{base_path2}/{filename}'
        
        return id, compare_images(image1, image2, prompt, model, temperature)
    except Exception as e:
        print(f"Error processing id {id}: {e}")
        return id, None

def parallel_process_comparisons(base_path1, base_path2, id_to_image_id, id_to_prompt, image_extension, model, temperature, num_images=100, max_workers=4):
    results = {}
    
    # 获取实际存在的ID，按顺序取前num_images个
    available_ids = sorted(id_to_image_id.keys())
    
    if num_images > len(available_ids):
        num_images = len(available_ids)
        
    selected_ids = available_ids[:num_images]
    total_tasks = len(selected_ids)
    
    if total_tasks == 0:
        print("No valid IDs found")
        return []
    
    print(f"Processing {total_tasks} images (IDs: {selected_ids[0]} to {selected_ids[-1]})")
    
    process_func = partial(process_single_comparison, 
                          base_path1=base_path1, 
                          base_path2=base_path2,
                          id_to_image_id=id_to_image_id,
                          id_to_prompt=id_to_prompt,
                          image_extension=image_extension,
                          model=model,
                          temperature=temperature)
    
    with tqdm(total=total_tasks, desc="Processing") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(process_func, id): id for id in selected_ids}
            
            for future in concurrent.futures.as_completed(future_to_id):
                id, result = future.result()
                results[id] = result
                
                # 更新进度条
                current_results = [r for r in results.values() if r in ['0', '1', '2']]
                if current_results:
                    ones_ratio = current_results.count('1') / len(current_results)
                    zeros_ratio = current_results.count('0') / len(current_results)
                    twos_ratio = current_results.count('2') / len(current_results)
                    pbar.set_description(f"Processed: {len(results)}/{total_tasks} | 1s: {ones_ratio:.2%}, 0s: {zeros_ratio:.2%}, 2s: {twos_ratio:.2%}")
                
                pbar.update(1)
    
    ordered_results = [results.get(id) for id in selected_ids]
    
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'comparison_results_{temperature}_{num_images}_{current_time}.json', 'w') as f:
        json.dump({
            'results': ordered_results,
            'processed_ids': selected_ids,
            'details': {'base_path1': base_path1, 'base_path2': base_path2}
        }, f, indent=4)
    
    return ordered_results

def main():
    BASE_PATH1 = '/nas/datasets/jinc/AHS/sd3_5'
    BASE_PATH2 = '/nas/datasets/jinc/AHS/sd3a_5'
    COCO_FILE = '/nas/datasets/coco/annotations/captions_val2017.json'
    IMAGE_EXTENSION = '.png' 
    MODEL = 'gpt-4o'
    TEMPERATURE = 0.0 
    NUM_IMAGES = 100000 
    MAX_WORKERS = 50
    
    print("Loading COCO data...")
    id_to_image_id, id_to_prompt = load_coco_data(COCO_FILE)
    print(f"Loaded {len(id_to_image_id)} annotations")
    
    # 显示一些示例映射
    available_ids = sorted(id_to_image_id.keys())
    sample_ids = available_ids[:3]
    print(f"Available IDs range: {available_ids[0]} to {available_ids[-1]}")
    print("Sample mappings:")
    for sample_id in sample_ids:
        image_id = id_to_image_id[sample_id]
        prompt = id_to_prompt[sample_id][:50] + "..." if len(id_to_prompt[sample_id]) > 50 else id_to_prompt[sample_id]
        filename = get_image_filename(image_id, IMAGE_EXTENSION)
        print(f"  ID {sample_id} -> image_id {image_id} -> {filename}")
        print(f"    Prompt: {prompt}")
    
    # 执行比较
    results = parallel_process_comparisons(
        base_path1=BASE_PATH1,
        base_path2=BASE_PATH2,
        id_to_image_id=id_to_image_id,
        id_to_prompt=id_to_prompt,
        image_extension=IMAGE_EXTENSION,
        model=MODEL,
        temperature=TEMPERATURE,
        num_images=NUM_IMAGES,
        max_workers=MAX_WORKERS
    )
    
    # 统计结果
    valid_results = [r for r in results if r in ['0', '1', '2']]
    if valid_results:
        final_ratio = valid_results.count('1') / len(valid_results)
        print("\nFinal Results Summary:")
        print(f"Total comparisons: {len(results)}")
        print(f"Valid comparisons: {len(valid_results)}")
        print(f"Final ratio of 1s: {final_ratio:.2%}")
        print(f"Number of 0s: {valid_results.count('0')}")
        print(f"Number of 1s: {valid_results.count('1')}")
        print(f"Number of 2s: {valid_results.count('2')}")
        print(f"Number of None/Failed: {results.count(None)}")

if __name__ == "__main__":
    main()