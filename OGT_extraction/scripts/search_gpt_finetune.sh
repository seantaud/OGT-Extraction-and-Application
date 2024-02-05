#!/bin/bash

date
echo "searching finetuned GPT"
date

# Define the parameter ranges to search
per_device_train_batch_size_values=(4 8)
learning_rate_values=(5e-2 1e-2 5e-3 7e-3 5e-4)

# Set the flag for enabling fp16

method=Finetune
# model_name_or_path=PrLM/BioGPT-Large
model_name_or_path=PrLM/biogpt
# model_name_or_path=PrLM/BioMedLM

# Define the function to run run_ds_zero2.py with given parameters
run_search() {
  python find_best_generation_paras.py \
    --method "$1" \
    --model_name_or_path "$2" \
    --per_device_train_batch_size "$3" \
    --learning_rate "$4"
}

# Perform grid search
for per_device_train_batch_size in "${per_device_train_batch_size_values[@]}"; do
  for learning_rate in "${learning_rate_values[@]}"; do
    # Build the command with the current parameter values
    command="run_search $method $model_name_or_path $per_device_train_batch_size $learning_rate"
    
    # Run the command
    eval "$command"
  done
done



