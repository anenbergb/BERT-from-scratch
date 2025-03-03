from typing import Optional
import torch
import os

from datasets import concatenate_datasets, load_dataset, load_from_disk
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
        input_ids = []
        for example in examples:
            pass_through = {key: example[key] for key in self.pass_through_keys}
            pass_through["original_input_ids"] = example["input_ids"].copy()
            pass_through_examples.append(pass_through)
            input_ids.append({"input_ids": example["input_ids"]})

        batch = {**default_data_collator(pass_through_examples, return_tensors="pt"), **self.collator(examples)}
        return batch


def prepare_pretraining_dataset(
    tokenizer,
    token_sequence_length: int = 128,
    sample_limit: int = 0,
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

    tokenized_dataset = dataset.map(
        wrap_tokenize_and_chunk(tokenizer, token_sequence_length), batched=True, remove_columns=dataset.column_names
    )
    return tokenized_dataset


def load_pretraining_dataset(
    tokenizer,
    dataset_cache_path: Optional[str] = None,
    token_sequence_length: int = 128,
    sample_limit: int = 0,
):
    if dataset_cache_path is not None and os.path.exists(dataset_cache_path):
        dataset = load_from_disk(dataset_cache_path)
    else:
        dataset = prepare_pretraining_dataset(tokenizer, token_sequence_length, sample_limit)
        if dataset_cache_path is not None:
            dataset.save_to_disk(dataset_cache_path)
    return dataset
