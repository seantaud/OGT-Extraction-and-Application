import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.ops.adam import DeepSpeedCPUAdam
import deepspeed

from dataset import load_gqa_dataset, preprocess_data, compute_metrics
from utils import apply_lora, print_trainable_parameters
from arguments import parse_arguments
from get_trainer import get_trainer, CastOutputToFloat 


def main():
    args = parse_arguments()

        # Load and preprocess the dataset
    DataFile = {
        'train': 'data/OGT_QA_train.json',
        'validation': 'data/OGT_QA_validation.json',
        'test': 'data/OGT_QA_dev.json'
    }

    tokenizer = AutoTokenizer.from_pretrained("PrLM/BioMedLM")
    if tokenizer.sep_token_id is None:
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.sep_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_datasets = load_gqa_dataset(DataFile)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        "PrLM/BioMedLM",
        device_map='auto',
    )

    # Preprocess train, validation, and test datasets
    train_data_processed = preprocess_data(raw_datasets['train'], tokenizer)
    valid_data_processed = preprocess_data(raw_datasets['validation'], tokenizer)
    test_data_processed = preprocess_data(raw_datasets['test'], tokenizer)

    # Convert processed data to GPT input format
    train_data_gpt = convert_to_gpt_format(train_data_processed, tokenizer)
    valid_data_gpt = convert_to_gpt_format(valid_data_processed, tokenizer)
    test_data_gpt = convert_to_gpt_format(test_data_processed, tokenizer)

    # Apply LoRA
    config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=True
    )

    model = apply_lora(model, config)
    print_trainable_parameters(model)

    # Convert processed train and validation data to GPT input format
    train_data = convert_to_gpt_format(train_data_gpt)
    valid_data = convert_to_gpt_format(valid_data_gpt)
    test_data = convert_to_gpt_format(test_data_gpt)

    # Initialize DeepSpeed engine
    ds_config = {
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": DeepSpeedCPUAdam,
            "params": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "fp16": {
            "enabled": args.fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
    }

    ds_engine, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters(),
                                              config=ds_config)

    # Get the trainer
    trainer = get_trainer(ds_engine.module, train_data, valid_data, tokenizer, args, compute_metrics)

    # Train the model
    trainer.train()

    # Evaluate on the test set
    test_results = trainer.evaluate(test_data)

    print("Test Set Results:")
    print(f"F1 Score: {test_results['eval_f1']}")
    print(f"Exact Match Score: {test_results['eval_exact_match']}")

