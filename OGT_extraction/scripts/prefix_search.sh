#!/bin/bash
module load anaconda/2020.11
source activate peft

date
echo "data process"
# python OGT_QA_datasets.py
date

# Define the parameter ranges to search
psl_values=(16 32 64)
per_device_train_batch_size_values=(4 )
learning_rate_values=(5e-2 1e-2 5e-3 7e-3 5e-4 )

# Set the flag for enabling fp16
method=prefix
# model_name_or_path=PrLM/BioGPT-Large
# model_name_or_path=PrLM/biogpt
model_name_or_path=PrLM/BioMedLM

# Define the function to run run_ds_zero2.py with given parameters
run_prefix() {
  python run_prefix.py \
    --method "$1" \
    --model_name_or_path "$2" \
    --psl "$3" \
    --per_device_train_batch_size "$4" \
    --learning_rate "$5" 
}

# Perform grid search

for psl in "${psl_values[@]}"; do
  for per_device_train_batch_size in "${per_device_train_batch_size_values[@]}"; do
    for learning_rate in "${learning_rate_values[@]}"; do
      # Build the command with the current parameter values
      command="run_prefix $method $model_name_or_path $psl $per_device_train_batch_size $learning_rate"
      
      # Run the command
      eval "$command"
      # python scripts/search_best_gpt.py "checkpoints/${model_name}/${method}/"
      # python scripts/search_failed_gpt.py "checkpoints/${model_name}/${method}/"
    done
  done
done


