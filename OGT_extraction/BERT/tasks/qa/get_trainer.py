import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback
)



from BERT.tasks.qa.dataset import SQuAD
from BERT.training.trainer_qa import QuestionAnsweringTrainer
from BERT.model.utils import get_model, TaskType

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, qa_args = args

    # Load the tokenizer
    model_name_or_path = model_args.model_name_or_path
    # Create output directory if it doesn't exist
    ckpt_dir = os.path.join("checkpoints", model_name_or_path.replace("PrLM/", ""))
    os.makedirs(ckpt_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
      model_name_or_path,
      num_labels=2,
      revision=model_args.model_revision,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
    )

    model = get_model(model_args, TaskType.QUESTION_ANSWERING, config, fix_model=True)
    #print(model)

    dataset = SQuAD(tokenizer, data_args, training_args, qa_args, ckpt_dir)
    training_args.eval_accumulation_steps = 2
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        eval_examples=dataset.eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        post_process_function=dataset.post_processing_function,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=dataset.compute_metrics
    )

    return trainer, dataset.predict_dataset,dataset.predict_examples


