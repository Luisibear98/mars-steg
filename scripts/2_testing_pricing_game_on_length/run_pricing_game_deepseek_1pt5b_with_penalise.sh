#/bin/bash

python -m \
    mars_steg.train \
    experiments/c_length_penalisation/1_collusion.yaml \
    mars_steg/configs/config_deepseek_1pt5b.yaml \
    scripts/2_testing_pricing_game_on_length/with_penalise_overwrite.yaml \
    mars_steg/configs/prompt/price_training_deepseek.yaml
