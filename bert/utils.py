import torch
from torch.optim import AdamW


def configure_optimizer(model, weight_decay=0.01, learning_rate=2e-5):
    """
    mirrors the original BERT optimization strategy from the BERT paper.
    Weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1200

    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def get_lr_scheduler(optimizer, lr_warmup_iters, max_train_iters):
    # Define full BERT-like schedule: warm-up (10k steps) + decay (990k steps)
    # Note, that for AdamW optimizer, the scheduler.step() is called twice,
    # once per param group.
    def schedule(step):
        warmup_steps = min(max(0, lr_warmup_iters), max_train_iters)
        total_decay_steps = max_train_iters - warmup_steps
        step = max(0, min(max_train_iters, step))
        if step < warmup_steps:
            # Warm-up: 0 to 1e-4 over 10,000 steps
            # LR will be 0 for the very first iteration
            return step / warmup_steps
        elif total_decay_steps == 0:
            return 1.0
        else:
            # Decay: 1e-4 to 0 over 990,000 steps
            decay_step = step - warmup_steps
            return max(0.0, 1.0 - decay_step / total_decay_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
