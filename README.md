# Look-Ahead and Look-Back Flows: Training-Free Image Generation with Trajectory Smoothing

## Install Prerequisites

```
conda env create -f environment.yml
```

## Download Datasets
```angular2html
python scripts/download_coco17.py
python scripts/download_cub200.py
python scripts/download_flickr30k.py
```

For COCO17 and Flickr30K, we use the CLIP model released by OpenAI to evaluate CLIPScores for each image-caption pair. Then, we use the caption with the best CLIPScore for inference and evaluation purposes, i.e.,

```bash
python scripts/measure_clipscore.py
python scripts/extract_one_caption_to_txt_best_clipscore.py
```

## Login to huggingface
```
huggingface-cli login
```

## Sampling and Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_flickr30k.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_coco17.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_cub200.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_lookahead_flickr30k.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_lookahead_coco17.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_lookahead_cub200.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_lookback_flickr30k.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_lookback_coco17.sh
CUDA_VISIBLE_DEVICES=0 bash inference_sd35_lookback_cub200.sh
```

We also provide the code of two baselines: [AFloPS](AFloPS) and [Self-Guidance](Self-Guidance).