#!/bin/bash

conda create -n selfguidance python=3.11
conda activate selfguidance

pip install torch torchvision diffusers accelerate transformers
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/boomb0om/text2image-benchmark
pip install hpsv2

# Install the package without dependencies
pip install -e . --no-deps
