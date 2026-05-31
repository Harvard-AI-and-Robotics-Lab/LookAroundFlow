#!/bin/bash

source activate base
conda activate trajectory_smoothing

HUGGINGFACE_TOKEN=''
BATCH_SIZE=2
NUM_PROMPTS_PER_RUN=4
NUM_PROMPTS_PER_RUN=1
IMAGE_WIDTH=512
IMAGE_HEIGHT=512

SCHEDULER="flow_euler"
INFERENCE_STEPS=25
GUIDANCE=7

ORG_IMAGE_DIR="datasets/cub_200/test"
PROMPT_DIR="datasets/cub_200/test"
BASE_OUTPUT_DIR="generated_images_cub200"

MODEL_PATH=(
            'output/cub200_pretrained_sd35/'
            )

LAMBDA=0.1

OUTPUT_DIR="${BASE_OUTPUT_DIR}_lookback_lmb${LAMBDA}_step${INFERENCE_STEPS}_g${GUIDANCE}"
GEN_IMG_DIR="${MODEL_PATH}/${OUTPUT_DIR}"

python inference_sd35_lookback.py \
    --prompt_dir=${PROMPT_DIR} \
    --model_path=${MODEL_PATH} \
    --output_dir=${OUTPUT_DIR} \
    --batch_size=${BATCH_SIZE} \
    --num_prompts_per_run=${NUM_PROMPTS_PER_RUN} \
    --image_width=${IMAGE_WIDTH} \
    --image_height=${IMAGE_HEIGHT} \
    --num_inference_steps=${INFERENCE_STEPS} \
    --guidance_scale=${GUIDANCE} \
    --scheduler=${SCHEDULER} \
    --seed 227 \
    --lookback_enabled true \
    --lookback_gamma_max 0.5 \
    --lookback_lambda_blend ${LAMBDA} \
    --lookback_snr_midpoint 0.0 

python evaluate_on_image_generation.py \
    --prompt_dir=${PROMPT_DIR} \
    --gt_img_dir=${ORG_IMAGE_DIR} \
    --gen_img_dir=${GEN_IMG_DIR}

python evaluate_with_blip_caption.py \
    --caption_folder=${PROMPT_DIR} \
    --image_folder=${GEN_IMG_DIR}

python assemble_performance_scores.py \
    --csv_file="${MODEL_PATH}/${OUTPUT_DIR}_img_quality.txt" \
    --json_file="${MODEL_PATH}/${OUTPUT_DIR}_caption_results.json" \
    --output="${MODEL_PATH}/${OUTPUT_DIR}_perf.txt"
