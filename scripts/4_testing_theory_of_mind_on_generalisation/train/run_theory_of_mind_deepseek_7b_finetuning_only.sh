#/bin/bash

python -m \
   mars_steg.train_generalisation \
    experiments/d_generalisation/1_theory_of_mind_objects.yaml \
    mars_steg/configs/config_deepseek_7b.yaml \
    experiments/d_generalisation/config_override_finetuning.yaml \
    mars_steg/configs/prompt/theory_of_mind_training_deepseek.yaml
