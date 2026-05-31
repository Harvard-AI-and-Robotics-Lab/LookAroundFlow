## Overview

This repository is the Official code for A-FloPS: Accelerating Diffusion Sampling with Adaptive Flow Path Sampler.
It contains:
   - `utils.py`: Implementation of **AFloPS** for DiT and **A‑Euler** for SDv3.5.  
   - `vis.ipynb`: Code for generating **Figure 1** in the paper.  
   - `vis_sd3.ipynb`: Code for generating **Figure 2** in the paper.  
   - `generate_distributed.py`: Generates results for **baseline samplers**.  
   - `generate_distributed_flops.py`: Generates results for **FloPS** and **AFloPS**.  
   - `gen_sd3.py`: Generates results for **SDv3.5**.  
   - `eval.py`: Modified from [openai/guided-diffusion](https://github.com/openai/guided-diffusion) to support folder inputs; used for evaluating DiT results.  
   - `eval_clip_ir.py`: Evaluates SDv3.5 results using CLIP and ImageReward (IR) metrics.  
   - `gpt.py`: LLM-assisted evaluation script (requires API key).  

---

## 1. Environment Setup

```bash
# Create a virtual environment
conda create -n aflops python=3.10
conda activate aflops

# Install PyTorch and torchvision matching your system
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip -r requirement.txt

```

------

## 2. Visualization

- **Figure 1**: Run `vis.ipynb`
- **Figure 2**: Run `vis_sd3.ipynb`

------

## 3. Sampling

### 3.1 Proposed Algorithms (FloPS / AFloPS)

```bash
torchrun --nproc_per_node=8 generate_distributed_flops.py \
    --sampler=aflops \
    --num_inference_steps=5 \
    --guidance_scale=1.5 \
    --mode=1 \  # 1 = AFloPS, 0 = FloPS
    --c_max=1 \
    --save_dir=./images
```

### 3.2 Baseline Samplers

```bash
torchrun --nproc_per_node=8 generate_distributed.py \
    --sampler=[unipc/ddim/dpmsolver] \
    --num_inference_steps=5 \
    --guidance_scale=1.5 \
    --save_dir=./images
```

### 3.3 SDv3.5

```bash
python gen_sd3.py \
    --prompt_dir=/coco_path/coco/annotations/captions_val2017.json \
    --device=cuda:0 \
    --sampler=aeuler \
    --num_inference_steps=5 \
    --save_dir=./images
```

------

## 4. Evaluation

### 4.1 Main Results & First Ablation

1. Download ImageNet reference batch (see [guided-diffusion/evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations)).
2. Run:

```bash
python eval.py \
    --ref_batch /path/to/reference/VIRTUAL_imagenet256_labeled.npz \
    --sample_batch /path/to/sample \
    --save_path /path/to/save/results
```

### 4.2 Second Ablation (CLIP & IR)

```bash
python eval_clip_ir.py \
    --caption_json /coco_path/coco/annotations/captions_val2017.json \
    --image_folder /generate_samples_path \
    --save_csv
```

### 4.3 Second Ablation (GPT Evaluation)

1. Edit the following variables in `gpt.py`:

```python
API_KEY = "your_api_key_here"
base_url = "YOUR_OPENAI_BASE_URL"
BASE_PATH1 = "/path/to/aeuler"
BASE_PATH2 = "/path/to/euler"
COCO_FILE = "/coco_path/coco/annotations/captions_val2017.json"
```

1. Run:

```bash
python gpt.py
```

## Citation
If you find this project useful in your research, please consider citing:  

```bibtex
@misc{jin2025aflopsacceleratingdiffusionsampling,
      title={A-FloPS: Accelerating Diffusion Sampling with Adaptive Flow Path Sampler}, 
      author={Cheng Jin and Zhenyu Xiao and Yuantao Gu},
      year={2025},
      eprint={2509.00036},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.00036}, 
}
