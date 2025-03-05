from typing import List, Optional, Any, Tuple
import torch
import os
from loguru import logger
import random

from datasets import concatenate_datasets, load_dataset, load_from_disk, DatasetDict
from transformers import default_data_collator, DataCollatorForWholeWordMask
from transformers import BertTokenizerFast, BertTokenizer


def wrap_tokenize_and_chunk(tokenizer, max_length=32):
    def tokenize_and_chunk(examples):
        # https://huggingface.co/docs/transformers/main/en/pad_truncation
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_overflowing_tokens=True,
        )
        result.pop("overflow_to_sample_mapping")
        return result

    return tokenize_and_chunk


def pretokenizer(tokenizer, text):
    normalizer = tokenizer.backend_tokenizer.normalizer
    pretokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    return pretokenizer.pre_tokenize_str(normalizer.normalize_str(text))


class TrainingCollator:
    def __init__(self, tokenizer, mask_lm_prob=0.15):
        self.pass_through_keys = ["token_type_ids", "attention_mask"]
        self.collator = DataCollatorForWholeWordMask(
            tokenizer, mlm=True, mlm_probability=mask_lm_prob, return_tensors="pt"
        )

    def __call__(self, examples):
        pass_through_examples = []
        for example in examples:
            pass_through = {key: example[key] for key in self.pass_through_keys}
            pass_through["original_input_ids"] = example["input_ids"].copy()
            pass_through_examples.append(pass_through)

        batch = {**default_data_collator(pass_through_examples, return_tensors="pt"), **self.collator(examples)}
        return batch


def prepare_pretraining_dataset(
    tokenizer,
    token_sequence_length: int = 128,
    sample_limit: int = 0,
    test_percent: float = 0.1,
    random_seed: int = 0,
):
    """
    sample_limit (int): limit the number of text samples to use for pre-training
    """
    if not isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
        raise ValueError(
            "BERT pre-training dataset preparation can only be " " performed for BertTokenizer-like tokenizers. "
        )
    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    # only keep the 'text' column
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
    assert bookcorpus.features.type == wiki.features.type
    dataset = concatenate_datasets([bookcorpus, wiki])
    if sample_limit > 0:
        dataset = dataset.select(range(sample_limit))

    dataset = dataset.train_test_split(test_size=test_percent, shuffle=True, seed=random_seed)

    tokenized_dataset = DatasetDict()
    for key, split_dataset in dataset.items():
        tokenized_dataset[key] = split_dataset.map(
            wrap_tokenize_and_chunk(tokenizer, token_sequence_length),
            batched=True,
            remove_columns=split_dataset.column_names,
        )
    return tokenized_dataset


def load_pretraining_dataset(
    tokenizer,
    dataset_cache_path: Optional[str] = None,
    token_sequence_length: int = 128,
    sample_limit: int = 0,
    test_percent: float = 0.1,
    random_seed: int = 0,
):
    if dataset_cache_path is not None and os.path.exists(dataset_cache_path):
        dataset = load_from_disk(dataset_cache_path)
    else:
        dataset = prepare_pretraining_dataset(tokenizer, token_sequence_length, sample_limit, test_percent, random_seed)
        if dataset_cache_path is not None:
            logger.info(f"Saving dataset to {dataset_cache_path}")
            dataset.save_to_disk(dataset_cache_path)
    return dataset


class DataCollatorForWholeWordMaskDeterministic(DataCollatorForWholeWordMask):
    def __init__(self, *args, random_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_seed = random_seed
        self.call_counter = 0

    def __call__(self, features, return_tensors=None):
        random.seed(self.random_seed + self.call_counter)
        self.call_counter += 1
        return super().__call__(features, return_tensors)

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels
