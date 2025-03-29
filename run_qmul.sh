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
    mars_steg.train_shortening \
    experiments/c_length_penalisation/1_collusion.yaml \
    mars_steg/configs/config_deepseek_14b.yaml \
    experiments/c_length_penalisation/overwrite_weight.yaml \
    mars_steg/configs/prompt/price_training_deepseek.yaml  # XXX: 20250212