import json
import os
from tqdm import tqdm
import numpy as np
import copy
import pandas as pd

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)

import evaluate
from OGT_QA_datasets import load_samples, load_examples, process_examples
from GPT.dataset import NEGATIVE_ANSWER, func_replace, preprocess_dataset
from GPT.utils import CastOutputToFloat, print_trainable_parameters
from GPT.arguments import parse_arguments
from torch.utils.data import DataLoader, ConcatDataset
from datasets import concatenate_datasets

from peft import PrefixTuningConfig, TaskType, get_peft_model


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parse_arguments()
    device = "cuda"
    method = args.method
    # assert method == "Prefix", f"{method}is not Prefix"

    set_seed(42)
    do_predict = True

    print('\n\n*************************************************************\n*************************************************************\n')
    print(args)
    print('\n*************************************************************\n*************************************************************\n\n')


    # Load the tokenizer
    model_name_or_path = args.model_name_or_path
    # Create output directory if it doesn't exist
    ckpt_dir = os.path.join("model-augments_is_7_384_3_g", model_name_or_path.replace("PrLM/", ""))
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
    max_length = 384 # 768
    if hasattr(tokenizer, "model_max_length"):
        test_length = tokenizer.model_max_length
    elif hasattr(tokenizer, "max_position_embeddings"):
        test_length = tokenizer.max_position_embeddings
    else:
        assert False, "max_length not set"

    test_length = min(test_length, 1000)-args.psl if method=='prefix' else min(test_length, 1000)

    # if True:
    if not os.path.exists(os.path.join(ckpt_dir, "train.json")) or not os.path.exists(
        os.path.join(ckpt_dir, "validation.json")
    ):
        raw_datasets = process_examples(
            load_examples(), tokenizer, max_length,test_length, ckpt_dir
        )
    else:
        raw_datasets = load_samples(ckpt_dir)

    raw_datasets['train'] = concatenate_datasets([raw_datasets['train'], raw_datasets['validation']])
    raw_datasets.pop('validation')

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

    do_training = True
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.exists(os.path.join(ckpt_dir, "best_model.pth")):
        if os.path.exists(os.path.join(ckpt_dir, "metrics.json")):
            with open(os.path.join(ckpt_dir, "metrics.json"), "r") as f:
                metrics = json.load(f)
                print(metrics)
            return
        else:
            do_training = False

    

    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [str(x) for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)
        for i in range(batch_size):
            # if i % 4==0:
            #     print(tokenizer.decode(model_inputs["input_ids"][i]))
            #     print(tokenizer.decode(labels["input_ids"][i]))
            if "biogpt" in model_name_or_path.lower():
                model_inputs["input_ids"][i] = model_inputs["input_ids"][i][1:]
                labels["input_ids"][i] = labels["input_ids"][i][1:] + [
                    tokenizer.eos_token_id
                ]

            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            # if len(sample_input_ids) > 1024:
            #     print("training too long")
            #     print(len(sample_input_ids))
            #     print(tokenizer.decode(model_inputs["input_ids"][i]))
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (
                max_length - len(sample_input_ids)
            ) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][:max_length]
            )
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][:max_length]
            )
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
            # if i % 4==0:
            #     print(tokenizer.decode(model_inputs["input_ids"][i]))
            #     print(tokenizer.decode(labels["input_ids"][i]))
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def val_preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [str(x) for x in examples[text_column]]
        model_inputs = tokenizer(inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][:max_length]
            )
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][:max_length]
            )
        return model_inputs

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

    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        batch_size=8,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )
    # train_dataset = processed_datasets["train"]
    eval_dataset = copy.copy(train_dataset)
    print(len(train_dataset))



    # eval_dataset = processed_datasets["validation"]

    test_dataset = dataset["test"].map(
        test_preprocess_function,
        batched=True,
        num_proc=1,
        batch_size=8,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on test dataset",
    )

    # test_dataset = processed_datasets["test"]

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=default_data_collator,
        batch_size=8
    )

    # print(next(iter(train_dataloader)))

    # creating model
    if ("biomedlm" in model_name_or_path.lower() or "biogpt-large" in model_name_or_path.lower()):
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, load_in_4bit=True
        )
    elif "biogpt" in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, load_in_8bit=True
        )


    peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=args.psl)

    if "biogpt" in model_name_or_path.lower():
        model.output_projection = CastOutputToFloat(model.output_projection)
    else:
        model.lm_head = CastOutputToFloat(model.lm_head)

    if method=='prefix':
        model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    # Dictionary to store metrics
    metrics = {"validation": [], "test": {}}
    

    if do_training:
        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # lr scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps, # 50
            num_training_steps=(len(train_dataloader) * args.num_train_epochs),
        )

        # training and evaluation
        model = model.to(device)

        best_f1 = float(0)
        patience = 3  # Number of epochs to wait for improvement  5
        early_stop_counter = 0  # Counter to keep track of epochs without improvement
        for epoch in range(args.num_train_epochs):
            model.train()
            print("Run training")
            gpu()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            model.eval()
            if epoch % 5 == 0:
                print("Run evaling")
                gpu()
                eval_loss = 0
                eval_preds = []
                for step, batch in enumerate(tqdm(eval_dataloader)):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model.generate(
                    **batch,
                    max_new_tokens=10,
                    do_sample = False,
                    num_beams= 1,
                    )

                    preds = outputs[:, max_length:].detach().cpu().numpy()
                    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                    eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
                    # eval_preds.extend(
                    #     tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                    # )


                true_labels = dataset["train"][label_column]
                assert len(eval_preds) == len(
                    true_labels
                ), f"{len(eval_preds)} != {len(true_labels)}"

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
                        raw_datasets["train"]["answers"],
                        raw_datasets["train"]["id"],
                    )
                ]
                predictions = [
                    {
                        "prediction_text": "" if p in NEGATIVE_ANSWER else p,
                        "id": str(i),
                        "no_answer_probability": 1.0 if NEGATIVE_ANSWER in p else 0.0,
                    }
                    for p, i in zip(eval_preds, raw_datasets["train"]["id"])
                ]

                squad_v2_metric = evaluate.load("./metric/squad_v2")
                results = squad_v2_metric.compute(
                    predictions=predictions, references=references
                )
                f1 = results["f1"]
                em = results["exact"]
                print(f1, em)
                metrics["validation"].append(
                    {"epoch": epoch, "eval_f1": float(f1), "eval_em": float(em)})
                # Check if evaluation loss has improved
            
                if f1 > best_f1:
                    best_f1 = f1
                    early_stop_counter = 0
                    # Save the best model's state dict
                    best_model_state_dict = model.state_dict()
                    # Save the best model's state dict
                    torch.save(
                            best_model_state_dict, os.path.join(ckpt_dir, "best_model.pth")
                    )
                else:
                    early_stop_counter += 1

            # Stop training if there's no improvement for the specified number of epochs
                if early_stop_counter >= patience:
                    print("Early stopping triggered. No improvement in evaluation loss.")
                    break
        
        model.cuda()
        # Save metrics to JSON file
        metrics_file = os.path.join(ckpt_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

    if do_predict:
        # Load the best model's state dict
        print("Run prediction")
        
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pth")))
        model.eval()
        test_preds = []
        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=10,
                    do_sample = False,
                    num_beams= 1,
                )

            preds = outputs[:, test_length:].detach().cpu().numpy()
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        true_labels = dataset["test"][label_column]
        assert len(test_preds) == len(
            true_labels
        ), f"{len(test_preds)} != {len(true_labels)}"

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
        em = results["exact"]
        print(f"Test F1 score: {f1:.2f}")
        print(f"Test EM score: {em:.2f}%")
        print(results)
        print(f"{test_preds[:10]=}")
        print(f"{true_labels[:10]=}")

        # Store test metrics in the dictionary
        metrics["test"]["f1"] = f1
        metrics["test"]["em"] = em

        pa = [f1, em, str(ckpt_dir)]
        df_new = pd.DataFrame(pa, index = ["f1", "em", "model"]).T
                                
        if os.path.exists('/hy-tmp/OGT_extraction/test_outputs.csv'):
            df_all = pd.read_csv('/hy-tmp/OGT_extraction/test_outputs.csv')
            df_all = pd.concat([df_all, df_new])
            df_all.to_csv('/hy-tmp/OGT_extraction/test_outputs.csv', index = False)
        else:
            df_new.to_csv('test_outputs.csv', index = False)

        # Save metrics to JSON file
        metrics_file = os.path.join(ckpt_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        test_df = raw_datasets["test"].remove_columns("answers").to_pandas()
        test_df[label_column] = test_preds
        # print(test_df[["answer", label_column]].sample(20))

        pred_df = test_df[["id", label_column]]
        pred_df.columns = ["id", "Label"]
        pred_file = os.path.join(ckpt_dir, "predictions.csv")
        pred_df.to_csv(pred_file, index=False)

def gpu():
    device = "cuda"
    gpu_memory_allocated = torch.cuda.memory_allocated(device)
    gpu_memory_cached = torch.cuda.memory_cached(device)
    
    print("已分配显存:", gpu_memory_allocated / 1024 ** 3, "GB")
    print("缓存显存:", gpu_memory_cached / 1024 ** 3, "GB")

if __name__ == "__main__":
    main()