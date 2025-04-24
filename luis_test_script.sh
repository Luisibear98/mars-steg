#!/bin/bash

CUDA_DEVICE=1
SCRIPT=./scripts/4_testing_theory_of_mind_on_generalisation/test/test_theory_of_mind_deepseek_7b_with_penalise.sh
MODEL_NAME=robust-field-29
FIXED_ARG=0
MAX=1592
STEP=24

for ((i=24; i<=$MAX; i+=$STEP)); do
    echo "Running with argument: $i"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $SCRIPT $MODEL_NAME $FIXED_ARG $i
done