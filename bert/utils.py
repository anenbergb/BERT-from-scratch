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


def decode_with_mask(tokenizer, input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special_tokens = [x for x in tokenizer.all_special_tokens if x != tokenizer.mask_token]
    filtered_tokens = [x for x in tokens if x not in special_tokens]
    text = tokenizer.convert_tokens_to_string(filtered_tokens)
    clean_text = tokenizer.clean_up_tokenization(text)
    return clean_text


def decode_pred_string(tokenizer, input_ids, mask_token_index, topk_tokens, k=0, with_prob=True):
    pred_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i, token_index in enumerate(mask_token_index.tolist()):
        pred_token_id = topk_tokens.indices[i][k].item()
        pred_token = tokenizer.convert_ids_to_tokens(pred_token_id)
        if with_prob:
            pred_prob = topk_tokens.values[i][k].item()
            pred_token = f"{pred_token}[{pred_prob:.1%}]"
        pred_tokens[token_index] = pred_token
    filtered_pred_tokens = [x for x in pred_tokens if x not in tokenizer.all_special_tokens]
    text = tokenizer.convert_tokens_to_string(filtered_pred_tokens)
    clean_text = tokenizer.clean_up_tokenization(text)
    return clean_text


def decode_batch(tokenizer, batch, token_logits, topk=2, with_prob=True):
    output = []
    batch_original_text = tokenizer.batch_decode(batch["original_input_ids"], skip_special_tokens=True)
    for batch_index in range(token_logits.size(0)):
        input_ids = batch["input_ids"][batch_index]
        # only decode those samples that have a mask token
        if torch.count_nonzero(input_ids == tokenizer.mask_token_id).item() == 0:
            continue

        decoded = {
            "text": batch_original_text[batch_index],
            "text_with_mask": decode_with_mask(tokenizer, input_ids),
        }

        mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[0]
        mask_token_logits = token_logits[batch_index, mask_token_index, :]
        mask_token_probs = torch.softmax(mask_token_logits, dim=1)
        topk_tokens = torch.topk(mask_token_probs, topk, dim=1)

        for k in range(topk):
            decoded[f"pred_top_{k+1}"] = decode_pred_string(
                tokenizer, input_ids, mask_token_index, topk_tokens, k, with_prob
            )
        output.append(decoded)
    return output
