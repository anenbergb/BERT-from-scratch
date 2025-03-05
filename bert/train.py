from dataclasses import dataclass, field
import argparse
import sys
import logging
import os
from loguru import logger
from tqdm import tqdm
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertTokenizerFast


# Huggingface
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from safetensors.torch import load_model


from bert.model import BertConfig, BertMLM
from bert.data import load_pretraining_dataset, TrainingCollator, TrainingCollatorForTest
from bert.utils import configure_optimizer, get_lr_scheduler, decode_batch, worker_init_fn, set_seeds


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
    checkpoint_iters: int = field(default=10000)
    evaluation_iters: int = field(default=10000)  # how often to evaluate the model

    test_percent: float = field(default=0.1)
    seed: int = field(default=0)
    mixed_precision: str = field(default="bf16")  # no for float32

    # Dataloading
    initial_seq_len_dataset_cache_path: str = field(default=None)
    max_seq_len_dataset_cache_path: str = field(default=None)

    train_batch_size: int = field(default=256)
    initial_seq_len_train_batch_size: int = field(default=128)
    max_seq_len_train_batch_size: int = field(default=24)

    val_batch_size: int = field(default=128)
    # reduce the number of samples in the dataset for debugging purposes
    dataset_sample_limit: int = field(default=0)  # 0 means no limit
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
    train_only_with_initial_seq_len: bool = field(default=False)


def load_tokenized_dataloader(config: TrainingConfig, select_initial_seq_len=True) -> DataLoader:
    tokenizer_fast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer_slow = BertTokenizer.from_pretrained("bert-base-uncased")

    if select_initial_seq_len:
        sequence_length = config.initial_sequence_length
        dataset_cache_path = config.initial_seq_len_dataset_cache_path
        batch_size = config.initial_seq_len_train_batch_size
    else:
        sequence_length = config.max_sequence_length
        dataset_cache_path = config.max_seq_len_dataset_cache_path
        batch_size = config.max_seq_len_train_batch_size

    logger.info(f"Loading {sequence_length} length sequence dataloader")
    tokenized_dataset = load_pretraining_dataset(
        tokenizer_fast,
        dataset_cache_path=dataset_cache_path,
        token_sequence_length=sequence_length,
        sample_limit=config.dataset_sample_limit,
        test_percent=config.test_percent,
        random_seed=config.seed,
    )
    logger.info(f"Tokenized dataset\n{tokenized_dataset}")
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=TrainingCollator(tokenizer_slow, config.mask_lm_prob),
    )

    def build_val_dataloader():
        random_generator = torch.Generator()
        random_generator.manual_seed(config.seed)
        test_loader = DataLoader(
            tokenized_dataset["test"],
            batch_size=config.val_batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=config.num_workers,
            collate_fn=TrainingCollatorForTest(tokenizer_slow, config.mask_lm_prob, random_seed=config.seed),
            generator=random_generator,
        )
        return test_loader

    return train_loader, build_val_dataloader


class DataloaderIteratorInitialThenMax:
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
        torch.cuda.empty_cache()
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


class DataloaderIterator:

    def __init__(self, dataloader, accelerator, config: TrainingConfig):
        self.dataloader = dataloader
        self.config = config
        self.accelerator = accelerator

        self.initial_seq_gradient_accumulation_steps = (
            config.train_batch_size // config.initial_seq_len_train_batch_size
        )

        accelerator.print(
            f"By default, the model is pre-trained with sequences of length {config.initial_sequence_length} "
            f"for {config.initial_seq_len_training_fraction:.0%} of the steps and with sequences of length "
            f"{config.max_sequence_length} for the remaining steps.\n"
            f"In this 'train-only-with-initial-seq-len' mode, the model is pre-trained only with sequences of length "
            f"{config.initial_sequence_length} for the entire training.\n"
            f"Given the target batch size of {config.train_batch_size}, the gradient accumulation steps are "
            f"{self.initial_seq_gradient_accumulation_steps} for the {config.initial_sequence_length} length sequences "
            f"[batch_size {config.initial_seq_len_train_batch_size}]"
        )
        self.iter = iter(self.dataloader)
        self.step = config.start_iter

    def __iter__(self):
        return self

    def __next__(self):
        if self.step >= self.config.max_train_iters:
            raise StopIteration
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
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

    if config.train_only_with_initial_seq_len:
        train_dataloader, build_val_dataloader = load_tokenized_dataloader(config, select_initial_seq_len=True)
        train_dataloader = accelerator.prepare(train_dataloader)
        train_dataiterator = DataloaderIterator(train_dataloader, accelerator, config)
    else:
        initial_train_dataloader, _ = load_tokenized_dataloader(config, select_initial_seq_len=True)
        max_train_dataloader, build_val_dataloader = load_tokenized_dataloader(config, select_initial_seq_len=False)
        initial_train_dataloader, max_train_dataloader = accelerator.prepare(
            initial_train_dataloader, max_train_dataloader
        )
        train_dataiterator = DataloaderIteratorInitialThenMax(
            initial_train_dataloader, max_train_dataloader, accelerator, config
        )

    model = BertMLM(bert_config)
    criterion = nn.CrossEntropyLoss()  # ignore_index = -100
    optimizer = configure_optimizer(model, config.weight_decay, config.lr)
    lr_scheduler = get_lr_scheduler(optimizer, config.lr_warmup_iters, config.max_train_iters)

    model, optimizer, criterion, lr_scheduler = accelerator.prepare(model, optimizer, criterion, lr_scheduler)

    # TODO fix model saving names?
    # ONLY load the model weights from the checkpoint. Leave the optimizer and scheduler as is.
    if config.resume_from_checkpoint is not None and os.path.exists(config.resume_from_checkpoint):
        model_fpath = os.path.join(config.resume_from_checkpoint, "model.safetensors")
        assert os.path.exists(model_fpath), f"Model file {model_fpath} not found"
        accelerator.print(f"Loading model weights from {model_fpath}")
        weights_before = model.bias.detach().clone()
        load_model(
            accelerator._models[0],
            model_fpath,
            device=str(accelerator.device),
        )
        weight_after = model.bias.detach().clone()
        assert not torch.allclose(
            weights_before, weight_after
        ), "Model weights did not change after loading from checkpoint"

    if config.start_iter > 0:
        accelerator.print(f"Resuming training from iteration {config.start_iter}")
        for _ in range(config.start_iter):
            lr_scheduler.step()

    accelerator.print(f"Training from {config.start_iter} to {config.max_train_iters}")
    model.train()
    for step, batch in (
        progress_bar := tqdm(
            enumerate(train_dataiterator, start=config.start_iter),
            total=len(train_dataiterator),
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
                perplexity = torch.exp(loss)
            accelerator.backward(loss)  # accumulates gradients

        accelerator.clip_grad_norm_(model.parameters(), config.gradient_max_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        current_lr = lr_scheduler.get_last_lr()[0]
        logs = {
            "loss/train": loss.detach().item(),
            "perplexity/train": perplexity.detach().item(),
            "lr": current_lr,
            "sequence_length": batch["labels"].size(1),
        }
        accelerator.log(logs, step=step)
        progress_bar.set_postfix(**logs)

        if step > 0 and (step % config.checkpoint_iters == 0 or step == config.max_train_iters - 1):
            accelerator.save_state()

        if step > 0 and (step % config.evaluation_iters == 0 or step == config.max_train_iters - 1):
            torch.cuda.empty_cache()
            val_metrics = run_validation(
                accelerator,
                model,
                criterion,
                build_val_dataloader,
                limit_val_iters=config.limit_val_iters,
                global_step=step,
            )
            if accelerator.is_main_process:
                val_print_str = (
                    f"Validation metrics [Iteration {step}]: "
                    f"loss = {val_metrics['loss/val']:.2f}, perplexity = {val_metrics['perplexity/val']:.2f}"
                )
                accelerator.print(val_print_str)
                accelerator.log(val_metrics, step=step)

    accelerator.end_training()
    return 0


def run_validation(
    accelerator: Accelerator,
    model: nn.Module,
    criterion: nn.Module,
    build_val_dataloader: Callable,
    limit_val_iters: int = 0,
    global_step: int = 0,
):
    """
    NOTE: This function is written without consideration for distributed multi-GPU training.
    """
    val_dataloader = build_val_dataloader()
    val_dataloader = accelerator.prepare(val_dataloader)
    # val_dataloader.collate_fn.reset_call_counter()
    total_mask_tokens = 0
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for step, batch in tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader) if limit_val_iters == 0 else limit_val_iters,
            disable=not accelerator.is_local_main_process,
            desc="Validation",
        ):
            if limit_val_iters > 0 and step >= limit_val_iters:
                break
            with accelerator.autocast():
                logits = model(**batch)
                loss = criterion(logits.view(-1, bert_config.vocab_size), batch["labels"].view(-1))

            num_mask_tokens = torch.count_nonzero(batch["labels"] != -100).item()
            total_mask_tokens += num_mask_tokens
            total_loss += loss.detach().item() * num_mask_tokens

            if step == 0:
                batch = {k: v.detach().cpu() for k, v in batch.items()}
                logits = logits.detach().cpu()
                tokenizer = val_dataloader.collate_fn.tokenizer
                decoded_batch = decode_batch(tokenizer, batch, logits, topk=3, with_prob=True)
                logs = {}
                for batch_index, decoded in enumerate(decoded_batch):
                    decoded_text = ""
                    max_key_length = max(len(k) for k in decoded.keys())
                    for i, (k, v) in enumerate(decoded.items()):
                        spaces = " " * (max_key_length - len(k))
                        text = f"{k}: {spaces}{v}"
                        if i < len(decoded) - 1:
                            text += "\n"
                        decoded_text += text

                    logs[f"val-text-{batch_index:03d}"] = decoded_text
                accelerator.log(logs, step=global_step)

    avg_loss = total_loss / total_mask_tokens
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    val_metrics = {
        "loss/val": avg_loss,
        "perplexity/val": avg_perplexity,
    }
    torch.cuda.empty_cache()
    model.train()
    return val_metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Run Masked Language Model pre-training for BERT on the BookCorpus and English Wikipedia datasets.\n
This trainer is written without consideration for distributed multi-GPU training.
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
    parser.add_argument("--val-batch-size", type=int, default=64, help="Validation batch size")
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
        default=10000,
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
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--debug-dataset-sample-limit",
        type=int,
        default=0,
        help="Limit the number of samples in the dataset for debugging purposes",
    )
    parser.add_argument(
        "--train-only-with-initial-seq-len",
        action="store_true",
        help="Train only with the initial sequence length of 128. "
        "This will reduce the context window to 128 tokens for the entire training, "
        "resulting in a model that cannot handle long context windos of 512 tokens. "
        "This will significantly speed up training as compared to the training with the full 512 token context window.",
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
        dataset_sample_limit=args.debug_dataset_sample_limit,
        lr=args.lr,
        train_only_with_initial_seq_len=args.train_only_with_initial_seq_len,
    )
    bert_config = BertConfig(pre_layer_norm=args.pre_layer_norm)
    if args.train_only_with_initial_seq_len:
        bert_config.max_position_embeddings = config.initial_sequence_length
    sys.exit(train_mlm(config, bert_config))
