#!/bin/bash

source activate base
conda activate selfguidance

PROMPT_DIR="datasets/flickr30k/test"
OUTPUT_DIR="output/flickr30k_selfguidance_sd35/generated_images"

python gen_sd35.py \
    --seed 227 \
    --prompt_dir ${PROMPT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --resolution 512

PROMPT_DIR="datasets/coco17/validation"
OUTPUT_DIR="output/coco17_selfguidance_sd35/generated_images"

python gen_sd35.py \
    --seed 227 \
    --prompt_dir ${PROMPT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --resolution 512

PROMPT_DIR="datasets/cub_200/test/"
OUTPUT_DIR="output/cub200_selfguidance_sd35/generated_images"

python gen_sd35.py \
    --seed 227 \
    --prompt_dir ${PROMPT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --resolution 512