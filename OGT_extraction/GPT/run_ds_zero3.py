import os

from tqdm import tqdm
import numpy as np


import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
from deepspeed.ops.adam import DeepSpeedCPUAdam
import deepspeed
from accelerate import Accelerator
from datasets import load_metric
import evaluate
from dataset import load_gqa_dataset, process_dataset
from torch.utils.data import DataLoader
from utils import CastOutputToFloat, TorchTracemalloc, apply_lora, b2mb
from arguments import parse_arguments
from peft import LoraConfig,PrefixTuningConfig, TaskType, get_peft_model


if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parse_arguments()

    method = args.method
    set_seed(42)
    do_predict = True
    print(args)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    # Load and preprocess the dataset
    DataFile = {
        "train": "data/OGT_QA_train.json",
        "validation": "data/OGT_QA_validation.json",
        "test": "data/OGT_QA_dev.json",
    }
    

    # Load the tokenizer
    model_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.sep_token_id is None:
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.sep_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the datasets
    raw_datasets = load_gqa_dataset(DataFile)
    text_column = "input"
    label_column = "output"
    
    if hasattr(tokenizer,"model_max_length"):
        max_length = tokenizer.model_max_length
    elif hasattr(tokenizer,"max_position_embeddings"):
        max_length = tokenizer.max_position_embeddings
    else:
        assert False , "max_length not set"
    max_length = min(max_length,768)
    dataset = process_dataset(raw_datasets,tokenizer,max_length)

    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [str(x) for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
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
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def test_preprocess_function(examples):
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

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            test_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

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
        batch_size=args.per_device_train_batch_size,
        pin_memory=True,
    )

    # print(next(iter(train_dataloader)))

    # creating model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,load_in_4bit=True)
    print(model)
    
    if method=="LoRA":
        target_modules = None
    if "biogpt" in model_name_or_path.lower():
        target_modules = ["k_proj","v_proj","q_proj"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.r_value,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules
        )
    else :
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=args.psl
        )
    if "biogpt" in model_name_or_path.lower():
        model.output_projection = CastOutputToFloat(model.output_projection)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    )

    (
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
    )
    accelerator.print(model)

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    for epoch in range(args.num_train_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for index, batch in enumerate(tqdm(train_dataloader)):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(
            "GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin))
        )
        accelerator.print(
            "GPU Memory consumed at the end of the train (end-begin): {}".format(
                tracemalloc.used
            )
        )
        accelerator.print(
            "GPU Peak Memory consumed during the train (max-begin): {}".format(
                tracemalloc.peaked
            )
        )
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print(
            "CPU Memory before entering the train : {}".format(
                b2mb(tracemalloc.cpu_begin)
            )
        )
        accelerator.print(
            "CPU Memory consumed at the end of the train (end-begin): {}".format(
                tracemalloc.cpu_used
            )
        )
        accelerator.print(
            "CPU Peak Memory consumed during the train (max-begin): {}".format(
                tracemalloc.cpu_peaked
            )
        )
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        model.eval()
        eval_preds = []
        with TorchTracemalloc() as tracemalloc:
            for _, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                    )  # synced_gpus=True for DS-stage 3
                outputs = accelerator.pad_across_processes(
                    outputs, dim=1, pad_index=tokenizer.pad_token_id
                )
                preds = accelerator.gather_for_metrics(outputs)
                preds = preds[:, max_length:].detach().cpu().numpy()
                eval_preds.extend(
                    tokenizer.batch_decode(preds, skip_special_tokens=True)
                )

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(
            "GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin))
        )
        accelerator.print(
            "GPU Memory consumed at the end of the eval (end-begin): {}".format(
                tracemalloc.used
            )
        )
        accelerator.print(
            "GPU Peak Memory consumed during the eval (max-begin): {}".format(
                tracemalloc.peaked
            )
        )
        accelerator.print(
            "GPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print(
            "CPU Memory before entering the eval : {}".format(
                b2mb(tracemalloc.cpu_begin)
            )
        )
        accelerator.print(
            "CPU Memory consumed at the end of the eval (end-begin): {}".format(
                tracemalloc.cpu_used
            )
        )
        accelerator.print(
            "CPU Peak Memory consumed during the eval (max-begin): {}".format(
                tracemalloc.cpu_peaked
            )
        )
        accelerator.print(
            "CPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

        true_labels = dataset["validation"][label_column]
        assert len(eval_preds) == len(true_labels), f"{len(eval_preds)} != {len(true_labels)}"

        references = [{'answers': a, 'id': i} for a,i in zip(raw_datasets["validation"]["answers"],raw_datasets["validation"]["id"])]
        predictions = [{'prediction_text': p, 'id': i, 'no_answer_probability': 1. if p=="unanswerable" else 0.} for p,i in zip(eval_preds,raw_datasets["validation"]["id"])]
        accelerator.print( f"{len(eval_preds)} == {len(true_labels)}")
        squad_v2_metric = evaluate.load("squad_v2")
        results = squad_v2_metric.compute(predictions=predictions, references=references)
        f1 = results['f1']
        em =  results['exact']
        accelerator.print(f"Epoch={epoch} F1 score: {f1:.2f}")
        accelerator.print(f"Epoch={epoch} EM score: {em:.2f}%")
        accelerator.print(f"{eval_preds[:10]=}")
        accelerator.print(f"{true_labels[:10]=}")

    if do_predict:
        model.eval()
        test_preds = []
        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(
                outputs, dim=1, pad_index=tokenizer.pad_token_id
            )
            preds = accelerator.gather(outputs)
            preds = preds[:, max_length:].detach().cpu().numpy()
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        true_labels = dataset["test"][label_column]
        assert len(eval_preds) == len(true_labels), f"{len(eval_preds)} != {len(true_labels)}"

        references = [{'answers': a, 'id': i} for a,i in zip(raw_datasets["test"]["answers"],raw_datasets["test"]["id"])]
        predictions = [{'prediction_text': p, 'id': i, 'no_answer_probability': 1. if p=="unanswerable" else 0.} for p,i in zip(eval_preds,raw_datasets["test"]["id"])]

        squad_v2_metric = evaluate.load("squad_v2")
        results = squad_v2_metric.compute(predictions=predictions, references=references)
        f1 = results['f1']
        em =  results['exact']
        accelerator.print(f"Test F1 score: {f1:.2f}")
        accelerator.print(f"Test EM score: {em:.2f}%")
        accelerator.print(f"{eval_preds[:10]=}")
        accelerator.print(f"{true_labels[:10]=}")

        test_df = dataset["test"].to_pandas()
        test_df[label_column] = test_preds
        accelerator.print(test_df[[text_column, label_column]].sample(20))

        pred_df = test_df[["ID", label_column]]
        pred_df.columns = ["ID", "Label"]
        model_name = model_name_or_path.replace("PrLM/", "")
        os.makedirs(f"pred/{model_name}", exist_ok=True)
        pred_df.to_csv(f"data/{model_name}/predictions.csv", index=False)

    accelerator.wait_for_everyone()
