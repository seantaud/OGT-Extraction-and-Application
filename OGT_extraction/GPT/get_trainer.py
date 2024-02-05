import torch.nn as nn
from transformers import EarlyStoppingCallback, TrainingArguments, DataCollatorForLanguageModeling, Trainer
import transformers

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def get_trainer(model, train_data, valid_data, tokenizer, args, compute_metrics):
    model.lm_head = CastOutputToFloat(model.lm_head)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3,)

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            logging_steps=1,
            output_dir='outputs',
            save_strategy="epoch",
            evaluation_strategy="epoch",
            disable_tqdm=False,
            load_best_model_at_end=True,
            metric_for_best_model="f1",  # Use F1 score for early stopping
            greater_is_better=True,  # Larger F1 score is better

        ),
        callbacks=[early_stopping],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics,  # Add custom metric function
    )
    model.config.use_cache = False

    return trainer
