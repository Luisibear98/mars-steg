#!/bin/bash
#$ -cwd           # Set the working directory for the job to the current directory
#$ -l rocky
#$ -pe smp 8      # Request 8 core
#$ -l h_rt=1:0:0  # Request 1 hour runtime
#$ -l h_vmem=11G   # Request 8 x 11GB RAM
#$ -l gpu=1         # request 1 GPU

module load python/3.11.7-gcc-12.2.0

module load cuda

source mars_steg_env/bin/activate

python -m \
    mars_steg.train \
    experiments/b_sequential_price_game/1_collusion_with_neural_overseer.yaml \
    mars_steg/configs/config_deepseek_1pt5b.yaml \
    scripts/1_testing_with_and_without_penalisation/no_penalise_overwrite.yaml \
    mars_steg/configs/prompt/price_training_deepseek.yaml   # XXX: 20250212