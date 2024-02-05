#!/bin/bash

date
echo "data process"
date

# Define the parameter ranges to search
psl_values=(16 32 64)
per_device_train_batch_size_values=(8 4 16)
learning_rate_values=(7e-3 1e-2 5e-3)


method=P_tuning_v2
model_name_or_path=PrLM/biobert-large-cased-v1.1
model_name=${model_name_or_path#"PrLM/"}
task_name=qa
dataset_name=ogt_qa
num_train_epochs=20


# Define the function to run pt_v2.py with given parameters
run_BERT() {
  python run_BERT.py \
    --model_name_or_path "$1" \
    --dataset_name ogt_qa \
    --task_name qa \
    --do_train \
    --do_eval \
    --do_predict \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --greater_is_better True \
    --save_total_limit 2 \
    --prefix \
    --overwrite_output_dir \
    --per_device_train_batch_size "$2" \
    --learning_rate "$3" \
    --pre_seq_len "$4" \
    --num_train_epochs "$5" \
    --output_dir "$6" 
}

# Perform grid search

for psl in "${psl_values[@]}"; do
  for per_device_train_batch_size in "${per_device_train_batch_size_values[@]}"; do
    for learning_rate in "${learning_rate_values[@]}"; do
      # Define the output directory with the current parameter values

      output_dir="checkpoints/${model_name}/psl${psl}_batch${per_device_train_batch_size}_lr${learning_rate}/"

      

      # Build the command with the current parameter values
      command="run_BERT  $model_name_or_path $per_device_train_batch_size $learning_rate  $psl $num_train_epochs $output_dir"
      
      # Run the command
      eval "$command"
    done
  done
done