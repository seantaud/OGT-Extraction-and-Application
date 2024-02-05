import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="prefix", help="Method of PEFT tuning")
    parser.add_argument("--model_name_or_path", type=str, default="PrLM/biogpt", help="Model for PEFT tuning")
    parser.add_argument("--psl", type=int, default=8, help="Value for Prefix num_virtual_tokenst parameter")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument("--prefix", action="store_false", help="Enable prefix tuning")
    args = parser.parse_args()

    return args
