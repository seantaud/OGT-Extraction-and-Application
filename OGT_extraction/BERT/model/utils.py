from enum import Enum
import torch

from BERT.model.question_answering import (
    BertPrefixForQuestionAnswering,
    BertForQuestionAnswering,

)

from transformers import (
    AutoConfig,
)

class TaskType(Enum):
    QUESTION_ANSWERING = 1
    GENERATIVE_QUESTION_ANSWERING = 2

PREFIX_MODELS = {
    "bert": {
        TaskType.QUESTION_ANSWERING: BertPrefixForQuestionAnswering,
       
    },
    
}

PROMPT_MODELS = {
    "bert": {
        TaskType.QUESTION_ANSWERING: BertPrefixForQuestionAnswering,
        
    },
    
}

FT_MODELS = {
    "bert": {
        TaskType.QUESTION_ANSWERING: BertForQuestionAnswering,
    },
    
}

def get_model(
    model_args, 
    task_type: TaskType, 
    config: AutoConfig, 
    output_vocab_size = None,
    fix_model: bool = False
    ):
    
    config.output_vocab_size = output_vocab_size if output_vocab_size else config.vocab_size
    
    
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size
        
        model_class = PREFIX_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision
        )
    else:
        model_class = FT_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision
        )

        model_param = 0
            
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - model_param
        print('***** total param is {} *****'.format(total_param))

    return model
