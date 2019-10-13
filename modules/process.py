from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys

import torch

logger = logging.getLogger()

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from multiprocessing import Pool, cpu_count
from tqdm import tqdm_notebook


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir, set_type):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    def get_examples(self, data_dir, set_type):
        """See base class."""
        data = set_type + ".tsv"
        test_data = self._read_tsv(os.path.join(data_dir, data))
        return self._create_examples(test_data, set_type)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) == 4:
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                label = line[1]
                # print(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples[1:]


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    This function will truncate the longer sequence one token at the time.

    :param tokens_a: the first sentence
    :param tokens_b: the second sentence
    :param max_length: max length defined by the model params
    :return: trunctuated sequences
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return


def create_features(params, set_type):
    """
    Creates features if not yet exist in the data_dir, else it reads it and features.

    :params set_type: str, "train", "evaluation", "val" or "test"
    :return: tuples of (_features, _examples, _examples_len) for the set_type

    # BERT: cls + token_ids_0 + sep + token_ids_1 + sep
    # XLNet: token_ids_0 + sep + token_ids_1 + sep + cls
    # Roberta: cls + token_ids_0 + sep + sep + token_ids_1 + sep
    # XLM: cls + token_ids_0 + sep + token_ids_1 + sep
    """

    data_path = params['data_dir']
    max_seq_length = params['max_seq_len']
    tokenizer = params['tokenizer']
    model = params['model']

    name = f'{data_path}{set_type}_features_{max_seq_length}_{model}'

    if os.path.exists(name):
        print(f'Loading features from cached file {name}..')
        features = torch.load(name)
        print(f'Loading is finished from file {name}..')

    else:
        # Define how to to convert the examples
        cls_token_at_end = bool(params['model_type'] in ['xlnet'])
        sep_token_extra = bool(params['model_type'] in ['roberta'])
        cls_token_segment_id = 2 if params["model_type"] in ["xlnet"] else 1
        pad_on_left = bool(params["model_type"] in ["xlnet"])
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if params["model_type"] in ["xlnet"] else 0

        processor = BinaryClassificationProcessor()
        examples = processor.get_examples(data_path, set_type)
        examples_length = len(examples)
        label_list = processor.get_labels()

        label_map = {label: i for i, label in enumerate(label_list)}
        examples_for_processing = [(example, label_map, max_seq_length, tokenizer, \
                                    params["output_mode"], cls_token_at_end, \
                                    cls_token_segment_id, sep_token_extra, \
                                    pad_on_left, pad_token, pad_token_segment_id, sep_token_extra) for example in
                                   examples]

        process_count = cpu_count() - 1

        print(f'Preparing to convert {examples_length} examples..')
        print(f'Spawning {process_count} processes..')
        with Pool(process_count) as p:
            features = list(tqdm_notebook(p.imap(convert_example_to_feature, examples_for_processing), \
                                          total=examples_length))

        logger.info(f'Saving features into file: {name}')
        torch.save(features, name)

    # Save the params for running the code from REPL
    params_name = params['params_dict_dir']
    print(f'Adding {set_type}_example_length to the parameters dict and saving it in {params_name}')
    params[f'{set_type}_example_length'] = len(features)
    torch.save(params, params_name)

    return features, params


def convert_example_to_feature(example_row):
    """
    Returns the converted examples.
    First it tokenizes the sentences, next it places the [CLS] token at
    the beginning of the sequence and [SEP] between the sentences in a pair.

    It adds the input mask (contains 0 for padding tokens and 1 for real tokens)
    and the input ids (the id of the tokens).

    Finally it returns this as an InputFeatures class.

    # BERT: cls + token_ids_0 + sep + token_ids_1 + sep
         # if one sentence: cls + token_ids_0 + sep
    # XLNet: token_ids_0 + sep + token_ids_1 + sep + cls
         # if one sentence: token_ids_0 + sep + cls
    # Roberta: cls + token_ids_0 + sep + sep + token_ids_1 + sep
         # if one sentence: cls + token_ids_0 + sep
    # XLM: cls + token_ids_0 + sep + token_ids_1 + sep
         # if one sentence: cls + token_ids_0 + sep

    :param example_row: row of an input
    :return: InputFeatures class
    """

    example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, \
    cls_token_segment_id, sep_token_extra, \
    pad_on_left, pad_token, pad_token_segment_id, sep_token_extra = example_row
    tokens_a = tokenizer.tokenize(example.text_a)

    if sep_token_extra:
        tokens_a.append(tokenizer.sep_token)

    tokens_b = None

    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        if sep_token_extra:
            tokens_b.append(tokenizer.sep_token)
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        special_tokens_count = 4 if sep_token_extra else 3
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

    tokens = tokens_a + [tokenizer.sep_token]  # token_ids_0 + sep
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [tokenizer.sep_token]
        segment_ids += [1] * (len(tokens_b) + 1)

    if cls_token_at_end:  # Xlnet has cls token at the end
        tokens = tokens + [tokenizer.cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]  # This is the close token segment id for XLNet
    else:
        tokens = [tokenizer.cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)


def create_dataloader(features, params, evaluation=False):
    """
    Creates a dataloader from the example features

    :param features: list of Inputfeatures
    :param params: dictionary of model parameters
    :param evaluation: Boolean, True for the creation of the evaluation_dataloader, False otherwise
    :return: DataLoader class
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if evaluation:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=params["eval_batch_size"])
        return dataloader, all_label_ids
    else:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=params["train_batch_size"])

        return dataloader


def main():
    """This functions creates and dumps the features.
    It does not create the dataloaders, it will be created in the models_functions.py script."""

    params = torch.load(sys.argv[1])
    data_path = params['data_dir']
    max_seq_length = params['max_seq_len']
    model = params['model_type']
    print('Create features...')
    features, params = create_features(params, 'train')

    dataloader = create_dataloader(features, params, evaluation=False)
    dataloader_name = f'{data_path}dataloader_{max_seq_length}_{model}'
    torch.save(dataloader, dataloader_name)

    if params['do_eval']:
        print('Create evaluation features...')
        eval_features, params = create_features(params, 'evaluation')
        eval_dataloader, all_label_ids = create_dataloader(eval_features, params, evaluation=True)
        eval_dataloader_name = f'{data_path}eval_dataloader_{max_seq_length}_{model}'
        torch.save(eval_dataloader, eval_dataloader_name)
        torch.save(all_label_ids, f'{data_path}eval_label_ids')


if __name__ == '__main__':
    main()