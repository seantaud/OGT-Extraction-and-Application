#!/bin/bash

date
echo "searching prefix tuned GPT"
date

# Define the parameter ranges to search
psl_values=(16)
per_device_train_batch_size_values=(8)
learning_rate_values=(5e-4 )

# Set the flag for enabling fp16
method=prefix
# model_name_or_path=PrLM/BioGPT-Large
model_name_or_path=PrLM/biogpt

# Define the function to run find_best_generation_paras.py with given parameters
run_search() {
  python find_best_generation_paras.py \
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
      command="run_search $method $model_name_or_path $psl $per_device_train_batch_size $learning_rate"
      
      # Run the command
      eval "$command"
    done
  done
done


