from dataclasses import dataclass, field
import argparse
import sys
import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast


# Huggingface
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from safetensors.torch import load_model


from bert.model import BertConfig, BertMLM
from bert.data import load_pretraining_dataset, TrainingCollator
from bert.utils import configure_optimizer


@dataclass
class TrainingConfig:
    output_dir: str
    overwrite_output_dir: bool = field(default=True)  # overwrite the old model
    start_epoch: int = field(default=0)
    resume_from_checkpoint: str = field(default=None)
    epochs: int = field(default=100)
    limit_train_iters: int = field(default=0)
    limit_val_iters: int = field(default=0)
    checkpoint_total_limit: int = field(default=3)
    checkpoint_epochs: int = field(default=1)
    save_image_epochs: int = field(default=1)
    eval_epochs: int = field(default=1)  # how often to evaluate the model

    seed: int = field(default=0)
    mixed_precision: str = field(default="bf16")  # no for float32

    # Dataloading
    initial_dataset_cache_path: str = field(default=None)
    max_dataset_cache_path: str = field(default=None)

    train_batch_size: int = field(default=128)
    val_batch_size: int = field(default=256)
    # reduce the number of samples in the dataset for debugging purposes
    dataset_sample_limit: int = 100000  # 0 means no limit
    num_workers: int = field(default=2)

    # Linear warmup + CosineAnnealingLR
    # 2e-4 for AdamW
    lr: float = field(default=2e-4)
    lr_warmup_epochs: int = field(default=5)
    lr_warmup_decay: float = field(default=0.01)
    lr_min: float = field(default=0.0)

    # Regularization and Augmentation
    # 0.01 for AdamW
    weight_decay: float = field(default=0.01)
    norm_weight_decay: float = field(default=0.0)
    gradient_max_norm: float = field(default=2.0)

    # MLM pretraining
    mask_lm_prob: float = field(default=0.15)
    # hard-coded to 80% of the time, the token is replaced with [MASK]
    # hard-coded to 10% of the time, the token is replaced with a random token
    # hard-coded to 10% of the time, the token is left unchanged
    initial_sequence_length: int = field(default=128)
    max_sequence_length: int = field(default=512)


def train_mlm(config: TrainingConfig, bert_config: BertConfig) -> int:
    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        # logging_dir
        automatic_checkpoint_naming=True,
        total_limit=config.checkpoint_total_limit,
        save_on_each_node=False,
        iteration=config.start_epoch,  # the current save iteration
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
        step_scheduler_with_optimizer=False,
        split_batches=False,
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(os.path.basename(config.output_dir))

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    assert tokenizer.vocab_size == bert_config.vocab_size

    tokenized_dataset = load_pretraining_dataset(
        tokenizer,
        dataset_cache_path=config.initial_dataset_cache_path,
        token_sequence_length=config.initial_sequence_length,
        sample_limit=config.dataset_sample_limit,
    )
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=TrainingCollator(tokenizer, config.mask_lm_prob),
    )
    model = BertMLM(bert_config).to("cuda")

    optimizer = configure_optimizer(model, config.weight_decay, config.lr)
    criterion = nn.CrossEntropyLoss()
    #  masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

    for i, batch in enumerate(train_dataloader):
        if i > 100:
            break
        batch = {k: v.to("cuda") for k, v in batch.items()}
        output = model(**batch)
        print(output.shape)

    return 0


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Run Masked Language Model pre-training for BERT on the BookCorpus and English Wikipedia datasets.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/bryan/ssd01/expr/bert_from_scratch/run01",
        help="Path to save the model",
    )
    parser.add_argument("--train-batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=600, help="Epochs")
    parser.add_argument("--lr-warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument(
        "--limit-train-iters",
        type=int,
        default=0,
        help="Limit number of training iterations per epoch",
    )
    parser.add_argument(
        "--limit-val-iters",
        type=int,
        default=0,
        help="Limit number of val iterations per epoch",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Start epoch, useful for resuming training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint folder that the training should resume from.",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=1,
        help="Frequency of evaluation in epochs",
    )
    parser.add_argument(
        "--initial-dataset-cache-path",
        type=str,
        default=None,
        help="Path to pre-tokenized dataset used for the first 90 percent of training.",
    )
    parser.add_argument(
        "--max-dataset-cache-path",
        type=str,
        default=None,
        help="Path to pre-tokenized dataset used for the final 10% percent of training.",
    )
    parser.add_argument(
        "--pre-layer-norm",
        action="store_true",
        help="Use pre-layer norm in the transformer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig()
    args = get_args()
    config = TrainingConfig(
        output_dir=args.output_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr_warmup_epochs=args.lr_warmup_epochs,
        limit_train_iters=args.limit_train_iters,
        limit_val_iters=args.limit_val_iters,
        start_epoch=args.start_epoch,
        resume_from_checkpoint=args.resume_from_checkpoint,
        eval_epochs=args.eval_epochs,
        initial_dataset_cache_path=args.initial_dataset_cache_path,
        max_dataset_cache_path=args.max_dataset_cache_path,
    )
    bert_config = BertConfig(pre_layer_norm=args.pre_layer_norm)
    sys.exit(train_mlm(config, bert_config))
