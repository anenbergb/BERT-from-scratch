import argparse
import os
import sys
from loguru import logger
import shutil

from transformers import BertTokenizerFast

from bert.data import load_pretraining_dataset


def save_tokenized_dataset(
    save_path: str,
    sequence_length: int = 128,
    dataset_sample_limit: int = 0,
    test_percent: float = 0.1,
    random_seed: int = 0,
    overwrite: bool = False,
) -> None:
    if os.path.exists(save_path) and overwrite:
        logger.warning("Overwriting existing dataset.")
        shutil.rmtree(save_path)
    elif os.path.exists(save_path):
        logger.warning(f"Dataset already exists at {save_path}. Skipping.")
        return

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    logger.info(f"Loading and tokenizing {sequence_length} length sequence dataset")
    load_pretraining_dataset(
        tokenizer,
        dataset_cache_path=save_path,
        token_sequence_length=sequence_length,
        sample_limit=dataset_sample_limit,
        test_percent=test_percent,
        random_seed=random_seed,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Precompute the tokenized dataset for the BERT model.
This script will save the tokenized dataset to disk.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory to save the tokenized dataset to.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Length of the token sequences.",
    )
    parser.add_argument(
        "--dataset-sample-limit",
        type=int,
        default=0,
        help="Limit the number of samples in the dataset (0 means no limit).",
    )
    parser.add_argument(
        "--test-percent",
        type=float,
        default=0.1,
        help="Percentage of the dataset to use for testing.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for dataset shuffling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing dataset if it exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(
        save_tokenized_dataset(
            save_path=args.save_path,
            sequence_length=args.sequence_length,
            dataset_sample_limit=args.dataset_sample_limit,
            test_percent=args.test_percent,
            random_seed=args.random_seed,
            overwrite=args.overwrite,
        )
    )
