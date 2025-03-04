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
from accelerate.utils import ProjectConfiguration, GradientAccumulationPlugin
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

    train_batch_size: int = field(default=256)
    initial_seq_len_train_batch_size: int = field(default=128)
    max_seq_len_train_batch_size: int = field(default=24)

    val_batch_size: int = field(default=256)
    # reduce the number of samples in the dataset for debugging purposes
    dataset_sample_limit: int = 100000  # 0 means no limit
    num_workers: int = field(default=0)

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

    initial_seq_len_training_fraction: float = field(default=0.9)


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


class DataloaderIterator:
    """
    128 tokens for the first 90% of steps, then 512 tokens for the last 10%.
    """

    def __init__(self, dataloader1, dataloader2, accelerator, config: TrainingConfig):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.config = config
        self.accelerator = accelerator

        self.initial_seq_gradient_accumulation_steps = (
            config.train_batch_size // config.initial_seq_len_train_batch_size
        )
        self.max_seq_gradient_accumulation_steps = config.train_batch_size // config.max_seq_len_train_batch_size

        self.initial_seq_length_training_frac = self.get_initial_seq_len_training_fraction()
        self.max_seq_len_start_iter = int(self.initial_seq_length_training_frac * config.max_train_iters)

        ### print some info
        accelerator.print(
            f"By default, the model is pre-trained with sequences of length {config.initial_sequence_length} "
            f"for {config.initial_seq_len_training_fraction:.0%} of the steps and with sequences of length "
            f"{config.max_sequence_length} for the remaining steps.\n"
            f"Given the target batch size of {config.train_batch_size}, the gradient accumulation steps are "
            f"{self.initial_seq_gradient_accumulation_steps} for the {config.initial_sequence_length} length sequences "
            f"[batch_size {config.initial_seq_len_train_batch_size}] and "
            f"{self.max_seq_gradient_accumulation_steps} for the {config.max_sequence_length} length sequences "
            f"[batch_size {config.max_seq_len_train_batch_size}].\n"
            f"As a result, the initial sequence length training fraction is rescaled from {config.initial_seq_len_training_fraction:.0%} "
            f"to {self.initial_seq_length_training_frac:.0%} to account for the difference in gradient accumulation steps.\n"
            f"The training will start with sequences of length {config.initial_sequence_length} for the first "
            f"{self.max_seq_len_start_iter} steps and then switch to sequences of length "
            f"{config.max_sequence_length} for the remaining steps."
        )
        self.iter = iter(self.get_dataloader(config.start_iter))
        self.step = config.start_iter

    def get_initial_seq_len_training_fraction(self):
        """
        Rescale the initial sequence length training fraction to account for the
        the difference in gradient accumulation steps required for due to batch size
        difference between the initial sequence length and max sequence length.
        """
        frac = self.config.initial_seq_len_training_fraction
        iga = self.initial_seq_gradient_accumulation_steps
        mga = self.max_seq_gradient_accumulation_steps
        return frac * iga / (frac * iga + (1 - frac) * mga)

    def get_dataloader(self, step):
        if step < self.max_seq_len_start_iter:
            self.accelerator.gradient_accumulation_steps = self.initial_seq_gradient_accumulation_steps
            return self.dataloader1
        else:
            self.accelerator.gradient_accumulation_steps = self.max_seq_gradient_accumulation_steps
            return self.dataloader2

    def __iter__(self):
        return self

    def __next__(self):
        if self.step >= self.config.max_train_iters:
            raise StopIteration
        if self.step == self.max_seq_len_start_iter:
            self.iter = iter(self.get_dataloader(self.step))
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.get_dataloader(self.step))
            batch = next(self.iter)
        self.step += 1
        return batch

    def __len__(self):
        return config.max_train_iters - config.start_iter


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
        gradient_accumulation_steps=2,  # will be set in DataLoaderIter
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

    dataloader_iter = DataloaderIterator(initial_train_dataloader, max_train_dataloader, accelerator, config)
    for step, batch in (
        progress_bar := tqdm(
            enumerate(dataloader_iter, start=config.start_iter),
            total=len(dataloader_iter),
            disable=not accelerator.is_local_main_process,
            desc="Training",
        )
    ):
        with accelerator.accumulate(model):  # accumulate gradients into model.grad attributes
            with accelerator.autocast():
                logits = model(**batch)  # (N,seq_len,vocab_size)
                # (N,seq_len,vocab_size) -> (N*seq_len, vocab_size), labels: (N,seq_len) -> (N*seq_len,)
                # all of the non-masked tokens are -100, so they are ignored in the loss calculation
                loss = criterion(logits.view(-1, bert_config.vocab_size), batch["labels"].view(-1))
            accelerator.backward(loss)  # accumulates gradients

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
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
    parser.add_argument(
        "--initial-seq-len-train-batch-size", type=int, default=128, help="Initial sequence length batch size"
    )
    parser.add_argument("--max-seq-len-train-batch-size", type=int, default=24, help="Max sequence length batch size")
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
        initial_seq_len_train_batch_size=args.initial_seq_len_train_batch_size,
        max_seq_len_train_batch_size=args.max_seq_len_train_batch_size,
        val_batch_size=args.val_batch_size,
        max_train_iters=args.max_train_iters,
        lr_warmup_iters=args.lr_warmup_iters,
        limit_val_iters=args.limit_val_iters,
        start_iter=args.start_iter,
        resume_from_checkpoint=args.resume_from_checkpoint,
        evaluation_iters=args.evaluation_iters,
        initial_seq_len_dataset_cache_path=args.initial_dataset_cache_path,
        max_seq_len_dataset_cache_path=args.max_dataset_cache_path,
    )
    bert_config = BertConfig(pre_layer_norm=args.pre_layer_norm)
    sys.exit(train_mlm(config, bert_config))
