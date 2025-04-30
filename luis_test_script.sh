#!/bin/bash

CUDA_DEVICE=0
SCRIPT=./scripts/4_testing_theory_of_mind_on_generalisation/test/test_theory_of_mind_deepseek_7b_with_penalise.sh
MODEL_NAME=MARS-STEGO_TRAIN_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_seed_42
FIXED_ARG=0
MAX=1592
STEP=96

for ((i=96; i<=$MAX; i+=$STEP)); do
    echo "Running with argument: $i"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $SCRIPT $MODEL_NAME $FIXED_ARG $i
done