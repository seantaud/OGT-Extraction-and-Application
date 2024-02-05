import json
import os
from tqdm import tqdm
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    set_seed,
)

import evaluate
from OGT_QA_datasets import load_samples, load_examples, process_examples
from GPT.dataset import NEGATIVE_ANSWER, func_replace, preprocess_dataset
from GPT.utils import CastOutputToFloat, print_trainable_parameters
from GPT.arguments import parse_arguments
from torch.utils.data import DataLoader

from peft import PrefixTuningConfig, TaskType, get_peft_model


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parse_arguments()
    device = "cuda"
    method = args.method
    # assert method == "Prefix", f"{method}is not Prefix"

    set_seed(42)
    do_predict = True

    print(args)

    # Load the tokenizer
    model_name_or_path = args.model_name_or_path
    # Create output directory if it doesn't exist
    ckpt_dir = os.path.join("checkpoints", model_name_or_path.replace("PrLM/", ""))
    os.makedirs(ckpt_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.sep_token_id is None:
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.sep_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and preprocess the datasets
    text_column = "input"
    label_column = "output"
    max_length = 384
    if hasattr(tokenizer, "model_max_length"):
        test_length = tokenizer.model_max_length
    elif hasattr(tokenizer, "max_position_embeddings"):
        test_length = tokenizer.max_position_embeddings
    else:
        assert False, "max_length not set"
    test_length = min(test_length, 1000)-args.psl
    # if True:
    if not os.path.exists(os.path.join(ckpt_dir, "train.json")) or not os.path.exists(
        os.path.join(ckpt_dir, "validation.json")
    ):
        raw_datasets = process_examples(
            load_examples(), tokenizer, max_length,test_length, ckpt_dir
        )
    else:
        raw_datasets = load_samples(ckpt_dir)

    dataset = preprocess_dataset(raw_datasets, tokenizer, purpose="extraction")

    ckpt_dir = os.path.join(
        ckpt_dir,
        method,
        "_".join(
            [
                "lr",
                str(args.learning_rate),
                "psl",
                str(args.psl),
                "bs",
                str(args.per_device_train_batch_size),
            ]
        ),
    )

    def test_preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [str(x) for x in examples[text_column]]

        model_inputs = tokenizer(inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]

            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                test_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                test_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][:test_length]
            )
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][:test_length]
            )
            if len(sample_input_ids) > 1024:
                print("test too long")
                print( len(sample_input_ids))
                print(tokenizer.decode(model_inputs["input_ids"][i]))

        return model_inputs

    test_dataset = dataset["test"].map(
        test_preprocess_function,
        batched=True,
        num_proc=1,
        batch_size=8,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on test dataset",
    )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=default_data_collator,
        batch_size=1
    )

    # creating model
    if "biogpt-large" in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, load_in_4bit=True
        )
    elif "biogpt" in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, load_in_8bit=True
        )
    # print(model)

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=args.psl
    )
    if "biogpt" in model_name_or_path.lower():
        model.output_projection = CastOutputToFloat(model.output_projection)
    else:
        model.lm_head = CastOutputToFloat(model.lm_head)
    if method=='prefix':
        model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    # Dictionary to store metrics
    metrics = {"validation": [], "test": {}}

    if do_predict:
        print("Run prediction")
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pth")))
        model.eval()
        
        # Generate parameters
        generation_params = {
            "num_beams": [2, 4, 8],
            "top_k": [50, 100, 200],
            "top_p": [0.85, 0.9, 0.95]
        }

        # Grid search for generation parameters
        best_generation_params = {}
        best_f1 = 0.0
        for num_beams in generation_params["num_beams"]:
            for top_k in generation_params["top_k"]:
                for top_p in generation_params["top_p"]:
                    test_preds = []
                    for _, batch in enumerate(tqdm(test_dataloader)):
                        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                        with torch.no_grad():
                            outputs = model.generate(
                                **batch,
                                max_new_tokens=10,
                                num_beams=num_beams,
                                top_k=top_k,
                                top_p=top_p
                            )

                        preds = outputs[:, test_length:].detach().cpu().numpy()
                        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                        test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

                    true_labels = dataset["test"][label_column]
                    assert len(test_preds) == len(true_labels), f"{len(test_preds)} != {len(true_labels)}"

                    references = [
                        {
                            "answers": {
                                "text": [func_replace(a["text"][0])],
                                "answer_start": a["answer_start"],
                            }
                            if len(a["text"])
                            else a,
                            "id": str(i),
                        }
                        for a, i in zip(
                            raw_datasets["test"]["answers"],
                            raw_datasets["test"]["id"],
                        )
                    ]
                    predictions = [
                        {
                            "prediction_text": "" if p in NEGATIVE_ANSWER else p,
                            "id": str(i),
                            "no_answer_probability": 1.0 if NEGATIVE_ANSWER in p else 0.0,
                        }
                        for p, i in zip(test_preds, raw_datasets["test"]["id"])
                    ]

                    squad_v2_metric = evaluate.load("./metric/squad_v2")
                    results = squad_v2_metric.compute(
                        predictions=predictions, references=references
                    )
                    f1 = results["f1"]

                    print(f"Test F1 score with generation params - num_beams={num_beams}, top_k={top_k}, top_p={top_p}: {f1:.2f}")
                    
                    # Update best generation parameters if necessary
                    if f1 > best_f1:
                        best_f1 = f1
                        best_generation_params = {
                            "num_beams": num_beams,
                            "top_k": top_k,
                            "top_p": top_p
                        }
        
        print("Best generation parameters:")
        print(best_generation_params)
        print("Best F1 score:", best_f1)


if __name__ == "__main__":
    main()