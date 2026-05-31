#!/bin/bash

source activate base
conda activate aflops

PROMPT_DIR="datasets/flickr30k/test"
OUTPUT_DIR="output/flickr30k_aflops_sd35/generated_images"

python gen_sd3_captions.py \
  --prompt_dir ${PROMPT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --num_inference_steps 25 \
  --guidance_scale 7 \
  --seed 227 \
  --method aeuler

PROMPT_DIR="datasets/coco17/validation"
OUTPUT_DIR="output/coco17_aflops_sd35/generated_images"

python gen_sd3_captions.py \
  --prompt_dir ${PROMPT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --num_inference_steps 25 \
  --guidance_scale 7 \
  --seed 227 \
  --method aeuler

PROMPT_DIR="datasets/cub_200/test/"
OUTPUT_DIR="output/cub200_aflops_sd35/generated_images"

python gen_sd3_captions.py \
  --prompt_dir ${PROMPT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --num_inference_steps 25 \
  --guidance_scale 7 \
  --seed 227 \
  --method aeuler