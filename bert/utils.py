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
