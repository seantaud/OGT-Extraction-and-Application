import json
import logging
import os
import sys
import numpy as np
from typing import Dict
import shutil

import datasets
import transformers
from transformers import set_seed, Trainer
from transformers import AutoTokenizer
from datasets import load_dataset, Features, Value, Sequence, DatasetDict, Dataset
from transformers.trainer_utils import get_last_checkpoint

from BERT.arguments import get_args

from BERT.tasks.utils import *

os.environ["WANDB_DISABLED"] = "true"

feat = Features(
        {
            "id": Value(dtype="int32", id=None),
            "title": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
            "context": Value(dtype="string", id=None),
            "synonym_description": Value(dtype="string", id=None),
            "answers": Sequence(
                feature={
                    "text": Value(dtype="string", id=None),
                    "answer_start": Value(dtype="int32", id=None),
                },
                length=-1,
                id=None,
            ),
        }
    )

def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        return

    elif isinstance(predict_dataset, dict):
        
        for dataset_name, d in predict_dataset.items():
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


def main(fn, split):
    args = get_args()

    model_args, data_args, training_args, _ = args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
    )
    pad_on_right = tokenizer.padding_side == "right"

    def prepare_eval_dataset(examples):
        # if self.version_2:
        examples['question'] = [s+q for s,q in zip(examples['synonym_description'],examples['question'])]
            
        tokenized = tokenizer(
                examples['question' if pad_on_right else 'context'],
                examples['context' if pad_on_right else 'question'],
                truncation='only_second' if pad_on_right else 'only_first',
                max_length=384,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized["offset_mapping"][i])
                ]
        return tokenized

    def load_exp():
        data_files = {
            "test": "data/journal_dev.json",
        }
        raw_datasets = load_dataset("json", data_files=data_files, features=feat, cache_dir="/path/to/nonexistent/directory")
        return raw_datasets

    raw_datasets = load_exp()
    predict_examples = raw_datasets["test"]
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_examples.map(
            prepare_eval_dataset,
            batched=True,
            batch_size=4,
            remove_columns=raw_datasets["test"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on test dataset",
            )    


    log_level = training_args.get_process_log_level()

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    training_args.output_dir = training_args.output_dir + fn.replace('.json', '_{}'.format(split))
    print(training_args.output_dir)

    os.makedirs(training_args.output_dir, exist_ok=True)
    shutil.copy("data/journal_dev.json", '{}/dev.json'.format(training_args.output_dir))

    result_path= os.path.join(training_args.output_dir,'all_results.json')
    if os.path.exists(result_path):
        f = open(result_path,"r")
        test_result = json.load(f)
                
        return 

    if data_args.task_name.lower() == "qa":
        assert data_args.dataset_name.lower() in QA_DATASETS
        from BERT.tasks.qa.get_trainer import get_trainer
        
    else:
        raise NotImplementedError('Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(42)
    trainer, _, _ = get_trainer(args)

    last_checkpoint = None

    if training_args.do_predict:
        #predict(trainer, predict_dataset)

        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    
if __name__ == '__main__':

    for fn in os.listdir('journals/'):

        with open('journals//{}'.format(fn), 'r') as file1:
            data1_str = file1.readlines()
        
        json_objects = []
        for line in data1_str:
            line = line.replace("},", "}")
            data = json.loads(line)
            json_objects.append(data)


        count = 0
        split = 0
        file2 = open('data/journal_dev.json', 'w')
        for item in json_objects:
            context = item['context']
            taxid = item['taxid']
            doi = item['doi']
            name_in_context = item['name_context']
            if name_in_context == '':
                continue

            name_in_context = name_in_context.replace(',', ', ')
            scientific_name = item['scientific_name']
            question = 'what is the optimal growth temperature of ' + scientific_name + '?'
            context = item['context']
            if scientific_name == name_in_context:
                synonyms_desc = ''
            else:
                synonyms_desc = scientific_name + ' is also known as ' + name_in_context + '. '

            id = count + 1
            item_appended = {
                        'answers': {
                            'text': [],
                            'answer_start': []
                        },
                        'context':  context,
                        'id': id,
                        'question': question,
                        'synonym_description': synonyms_desc,
                        'title': str(taxid) + '+' + scientific_name   
            }


            if count != 250:
                
                json.dump(item_appended, file2)
                file2.write('\n') 
                count += 1
                split += 1

            else:
                file2.close()    
                try:
                    main(fn, split)
                    count = 0
                    file2 = open('data/journal_dev.json', 'w')
                    json.dump(item_appended, file2)
                    file2.write('\n') 
                    count += 1
                    split += 1

                except TypeError as e:
                    print(f"An error occurred: {e}")

        file2.close()
        try:
            main(fn, split)

        except TypeError as e:
            print(f"An error occurred: {e}")
    