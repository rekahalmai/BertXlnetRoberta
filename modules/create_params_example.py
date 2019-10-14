""" This script creates a dict containing the parameters for a BERT classification.
Example parameter dictionaries can be seen in ../example_dict/ """

import torch


def add_tokenizer(params):
    from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                      XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                                      XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                      RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }

    config_class, model_class, tokenizer_class = MODEL_CLASSES[params['model_type']]

    config = config_class.from_pretrained(params['model'], num_labels=2, finetuning_task=params['task_name'])
    tokenizer = tokenizer_class.from_pretrained(params['model'])
    params["tokenizer"] = tokenizer
    return params


def main():
    task_name = 'bert_128'
    params = {
        'data_dir': 'data/',  # Directory of train.tsv, evaluation.tsv and test.tsv (optional)
        'params_dict_dir': f'{task_name}_params.json',  # We save the dict here. It will be overwritten with more info.
        'model_type': 'bert',  # Defines the model type
        'model': 'bert-base-cased',  # Defines the exact model
        'task_name': 'binary',
        'output_dir': f'outputs/{task_name}/',  # The output will be saved in the output/bert_128 directory
        'cache_dir': 'cache/',
        'do_train': True,  # Need to specify True for training
        'do_eval': True,  # Need to specify True for evaluation
        'evaluate_during_training': False,  # If True, evaluation during training -> necessite val.tsv in data/
        # Note: the training will be very long if it is chosenn True
        'fp16': False,  # If True, decrease precision of all calculation (needs apex package)
        'fp16_opt_level': 'O1',  # Precision level (only relevant if fp16 is defined True)
        'max_seq_len': 128,  # Max text taken for one sentence
        'output_mode': 'classification',
        'train_batch_size': 32,
        'eval_batch_size': 32,

        'gradient_accumulation_steps': 1,
        'num_train_epochs': 1,
        'weight_decay': 0,
        'learning_rate': 4e-5,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.06,
        'warmup_steps': 0,
        'max_grad_norm': 1.0,

        'logging_steps': 50,
        'evaluate_during_training': False,
        'save_steps': 2000,
        'eval_all_checkpoints': True,

        'overwrite_output_dir': False,
        'reprocess_input_data': True,
        'notes': 'Using IMDB dataset',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    params = add_tokenizer(params)
    # Save the dict in directort defined in the dict
    torch.save(params, params['params_dict_dir'])

    return


if __name__ == '__main__':
    main()