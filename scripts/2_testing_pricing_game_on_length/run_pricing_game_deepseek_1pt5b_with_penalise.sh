#/bin/bash

python -m \
   mars_steg.train_shortening \
    experiments/c_length_penalisation/1_collusion.yaml \
    mars_steg/configs/config_deepseek_1pt5b.yaml \
    experiments/c_length_penalisation/overwrite_weight.yaml \
    mars_steg/configs/prompt/price_training_deepseek.yaml
