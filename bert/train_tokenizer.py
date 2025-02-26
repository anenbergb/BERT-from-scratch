import argparse
import sys
from loguru import logger

from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset
from transformers import BertTokenizerFast


def batch_iterator(dataset, batch_size=10000):
    for i in tqdm(range(0, len(dataset), batch_size)):
        yield dataset[i : i + batch_size]["text"]
    
def train_tokenizer(save_dir: str, vocab_size: int = 30000, batch_size: int = 10000) -> int:
    logger.info(f"Training tokenizer with vocab size {vocab_size}")
    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column
    
    assert bookcorpus.features.type == wiki.features.type
    dataset = concatenate_datasets([bookcorpus, wiki])
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(dataset, batch_size=batch_size), vocab_size=vocab_size)
    logger.info(f"Saving tokenizer to {save_dir}")
    tokenizer.save_pretrained(save_dir)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Train a BERT tokenizer. The tokenizer will be saved to the output directory.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="../tokenizer-30k",
        help="Path to save the tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30000,
        help="Vocabulary size for the tokenizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for training the tokenizer",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    sys.exit(train_tokenizer(args.save_dir, args.vocab_size, args.batch_size))
