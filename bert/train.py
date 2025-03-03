from dataclasses import dataclass, field
import argparse
import sys
import logging
import os
from loguru import logger
from tqdm import tqdm

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
from bert.utils import configure_optimizer, get_lr_scheduler


@dataclass
class TrainingConfig:
    output_dir: str
    overwrite_output_dir: bool = field(default=True)  # overwrite the old model
    start_iter: int = field(default=0)
    resume_from_checkpoint: str = field(default=None)

    # Track training by iterations rather than epochs
    max_train_iters: int = field(default=1000000)
    limit_val_iters: int = field(default=0)
    checkpoint_total_limit: int = field(default=3)
    checkpoint_iters: int = field(default=5000)
    evaluation_iters: int = field(default=5000)  # how often to evaluate the model

    seed: int = field(default=0)
    mixed_precision: str = field(default="bf16")  # no for float32

    # Dataloading
    initial_seq_len_dataset_cache_path: str = field(default=None)
    max_seq_len_dataset_cache_path: str = field(default=None)

    initial_seq_len_train_batch_size: int = field(default=128)
    max_seq_len_train_batch_size: int = field(default=128)

    val_batch_size: int = field(default=256)
    # reduce the number of samples in the dataset for debugging purposes
    dataset_sample_limit: int = 100000  # 0 means no limit
    num_workers: int = field(default=2)

    # Linear warmup + CosineAnnealingLR
    # 2e-4 for AdamW
    lr: float = field(default=1e-4)
    lr_warmup_iters: int = field(default=10000)

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


def load_initial_seq_len_dataloader(config: TrainingConfig, tokenizer: BertTokenizerFast) -> DataLoader:
    logger.info(f"Loading initial sequence length dataloader. seq_len={config.initial_sequence_length}")
    tokenized_dataset = load_pretraining_dataset(
        tokenizer,
        dataset_cache_path=config.initial_seq_len_dataset_cache_path,
        token_sequence_length=config.initial_sequence_length,
        sample_limit=config.dataset_sample_limit,
    )
    logger.info(f"Tokenized dataset\n{tokenized_dataset}")
    return DataLoader(
        tokenized_dataset,
        batch_size=config.initial_seq_len_train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=TrainingCollator(tokenizer, config.mask_lm_prob),
    )


def load_max_seq_len_dataloader(config: TrainingConfig, tokenizer: BertTokenizerFast) -> DataLoader:
    logger.info(f"Loading max sequence length dataloader. seq_len={config.max_sequence_length}")
    tokenized_dataset = load_pretraining_dataset(
        tokenizer,
        dataset_cache_path=config.max_seq_len_dataset_cache_path,
        token_sequence_length=config.max_sequence_length,
        sample_limit=config.dataset_sample_limit,
    )
    logger.info(f"Tokenized dataset\n{tokenized_dataset}")
    return DataLoader(
        tokenized_dataset,
        batch_size=config.max_seq_len_train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=TrainingCollator(tokenizer, config.mask_lm_prob),
    )


def train_mlm(config: TrainingConfig, bert_config: BertConfig) -> int:
    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        # logging_dir
        automatic_checkpoint_naming=True,
        total_limit=config.checkpoint_total_limit,
        save_on_each_node=False,
        iteration=config.start_iter,  # the current save iteration
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
        step_scheduler_with_optimizer=False,
        split_batches=False,
        # TODO: make suree we can scale up gradient accumulation for the
        # max sequence length batches https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation
        gradient_accumulation_steps=2,  # accumulate up to 256 batch size
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(os.path.basename(config.output_dir))

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    assert tokenizer.vocab_size == bert_config.vocab_size

    initial_train_dataloader = load_initial_seq_len_dataloader(config, tokenizer)
    max_train_dataloader = load_max_seq_len_dataloader(config, tokenizer)
    model = BertMLM(bert_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = configure_optimizer(model, config.weight_decay, config.lr)
    lr_scheduler = get_lr_scheduler(optimizer, config.lr_warmup_iters, config.max_train_iters)

    model, optimizer, initial_train_dataloader, max_train_dataloader, criterion, lr_scheduler = accelerator.prepare(
        model, optimizer, initial_train_dataloader, max_train_dataloader, criterion, lr_scheduler
    )

    # ONLY load the model weights from the checkpoint. Leave the optimizer and scheduler as is.
    if config.resume_from_checkpoint is not None and os.path.exists(config.resume_from_checkpoint):
        model_fpath = os.path.join(config.resume_from_checkpoint, "model.safetensors")
        assert os.path.exists(model_fpath), f"Model file {model_fpath} not found"
        accelerator.print(f"Loading model weights from {model_fpath}")
        weights_before = model.module.bias.detach().clone()
        load_model(
            accelerator._models[0],
            model_fpath,
            device=str(accelerator.device),
        )
        weight_after = model.module.bias.detach().clone()
        assert not torch.allclose(
            weights_before, weight_after
        ), "Model weights did not change after loading from checkpoint"

    if config.start_iter > 0:
        accelerator.print(f"Resuming training from iteration {config.start_iter}")
        for _ in range(config.start_iter):
            lr_scheduler.step()

    def get_dataloader(step):
        #  128 tokens for the first 90% of steps, then 512 tokens for the last 10%.
        max_seq_len_start_iter = int(0.9 * config.max_train_iters)
        if step < max_seq_len_start_iter:
            return initial_train_dataloader
        else:
            return max_train_dataloader

    dataloader = iter(get_dataloader(config.start_iter))
    for step in (
        progress_bar := tqdm(
            range(config.start_iter, config.max_train_iters),
            disable=not accelerator.is_local_main_process,
            desc="Training",
        )
    ):
        if step == int(0.9 * config.max_train_iters):
            dataloader = iter(max_train_dataloader)
        try:
            batch = next(dataloader)
        except StopIteration:
            dataloader = iter(get_dataloader(step))
            batch = next(dataloader)

        optimizer.zero_grad()
        with accelerator.accumulate(model):
            with accelerator.autocast():
                logits = model(**batch)  # (N,seq_len,vocab_size)
                # (N,seq_len,vocab_size) -> (N*seq_len, vocab_size), labels: (N,seq_len) -> (N*seq_len,)
                # all of the non-masked tokens are -100, so they are ignored in the loss calculation
                loss = criterion(logits.view(-1, bert_config.vocab_size), batch["labels"].view(-1))
            accelerator.backward(loss)  # accumulates gradients

        optimizer.step()
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        logs = {
            "loss/train": loss.detach().item(),
            "lr": current_lr,
        }
        progress_bar.set_postfix(**logs)

        if step % config.evaluation_iters == 0 or step == config.max_train_iters - 1:
            # Evaluation
            pass

    return 0


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Run Masked Language Model pre-training for BERT on the BookCorpus and English Wikipedia datasets.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/bryan/ssd01/expr/bert_from_scratch/run01",
        help="Path to save the model",
    )
    parser.add_argument("--train-batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=256, help="Validation batch size")
    parser.add_argument("--max-train-iters", type=int, default=1000000, help="Maximum training iterations")
    parser.add_argument("--lr-warmup-iters", type=int, default=10000, help="Warmup iterations")
    parser.add_argument(
        "--limit-val-iters",
        type=int,
        default=0,
        help="Limit number of validation iterations",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=0,
        help="Start iteration, useful for resuming training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint folder that the training should resume from.",
    )
    parser.add_argument(
        "--evaluation-iters",
        type=int,
        default=5000,
        help="Frequency of evaluation in iterations",
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
        start_iter=args.start_iter,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_train_iters=args.max_train_iters,
        limit_val_iters=args.limit_val_iters,
        evaluation_iters=args.evaluation_iters,
        initial_seq_len_dataset_cache_path=args.initial_dataset_cache_path,
        max_seq_len_dataset_cache_path=args.max_dataset_cache_path,
    )
    bert_config = BertConfig(pre_layer_norm=args.pre_layer_norm)
    sys.exit(train_mlm(config, bert_config))
