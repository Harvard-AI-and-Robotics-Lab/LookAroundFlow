#!/bin/bash

source activate base
conda activate trajectory_smoothing


OPENAI_API_KEY=

PROMPT_DIR="datasets/coco17/validation"
GEN_IMAGE_DIR=(
            'output/generated_images_coco17_lookahead.json'
                )

for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));
do
    PARENT_FOLDER=$(dirname "${GEN_IMAGE_DIR[$i]}")
    FULL_PATH="${PARENT_FOLDER}/${OUTPUT_JSON}"
    python evaluate_CLAIR_with_generated_captions.py \
    --input ${GEN_IMAGE_DIR[$i]} \
    --api-key ${OPENAI_API_KEY} 
done