#/bin/bash
# e.g.:
# scripts/4_testing_theory_of_mind_on_generalisation/test_theory_of_mind_deepseek_7b_with_penalise.sh fluent-vortex-1285 0 0

export run_name=$1
export run_epoch=$2
export run_step_in_epoch=$3
export resume_code="${run_name}_model_${run_epoch}_step_${run_step_in_epoch}"  # e.g. "fluent-vortex-1285_model_0_step_0"

export target_path="experiments/d_generalisation/test_config_cache/1_theory_of_mind_test_${resume_code}.yaml"
cp experiments/d_generalisation/1_theory_of_mind_test_template.yaml $target_path
echo "load_lora_from_path_wandb: ${resume_code}" >> $target_path

python -m \
   mars_steg.train_generalisation \
   $target_path \
   mars_steg/configs/config_deepseek_7b.yaml \
   experiments/d_generalisation/config_override.yaml \
   mars_steg/configs/prompt/theory_of_mind_training_deepseek.yaml
