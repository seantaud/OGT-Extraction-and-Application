#!/bin/bash

# Define the parameter ranges to search

model_name_or_path=checkpoints/BioLinkBERT-large/finetune/psl32_batch16_lr5e-5/checkpoint-546
task_name=qa
dataset_name=ogt_qa
num_train_epochs=20


# Define the function to run pt_v2.py with given parameters
run_predict() {
  python run_predict.py \
    --model_name_or_path "$1" \
    --dataset_name ogt_qa \
    --task_name qa \
    --do_predict \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --greater_is_better True \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --output_dir "$2" 
}

# Perform grid search
output_dir="Result/predict_result_"
# Build the command with the current parameter values
command="run_predict  $model_name_or_path $output_dir"  
# Run the command
eval "$command"
