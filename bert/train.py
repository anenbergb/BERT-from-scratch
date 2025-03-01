from dataclasses import dataclass, field
import torch
from torch import nn


@dataclass
class TrainingConfig:

    # MLM pretraining
    mask_lm_prob: float = 0.15
    # hard-coded to 80% of the time, the token is replaced with [MASK]
    # hard-coded to 10% of the time, the token is replaced with a random token
    # hard-coded to 10% of the time, the token is left unchanged

    initial_sequence_length: int = 128
    max_sequence_length: int = 512
    batch_size: int = 32
