#/bin/bash

python -m \
    mars_steg.train \
    experiments/c_length_penalisation/1_collusion.yaml \
    mars_steg/configs/config_deepseek_1pt5b.yaml \
    mars_steg/configs/prompt/price_training_deepseek.yaml
