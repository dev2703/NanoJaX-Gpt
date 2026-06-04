"""
Training and model hyperparameters (NanoGPT-style).

Edit values here, then run:
  python train_tokenizer.py
  python prepare_data.py
  python train.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

#Data Loader
dataset = "Data/input.txt"
out_dir = "out"
tokenizer_path = f"{out_dir}/tokenizer.json"
tokens_path = f"{out_dir}/tokens.bin"

#Hyperparametrs
block_size = 1024
batch_size = 32
val_fraction = 0.1

vocab_size = 512
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True
max_seq_len = 1024
rope_theta = 10000.0
rms_eps = 1e-5
vocab_pad_multiple = 64

#optimizer and LR scheduling
learning_rate = 6e-4
max_iters = 5000
warmup_iters = 200
weight_decay = 0.1
grad_clip = 1.0
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-8


# Logging / eval

eval_interval = 250
log_interval = 10
eval_iters = 20
seed = 1337
ema_decay = 0.98

# BPE
special_tokens: list[str] = []


def pad_vocab_size(size: int, multiple: int = vocab_pad_multiple) -> int:
    return ((size + multiple - 1) // multiple) * multiple


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    max_seq_len: int
    rope_theta: float = rope_theta
    rms_eps: float = rms_eps
    dtype: jnp.dtype = jnp.float32
    dropout: float = dropout
    bias: bool = bias
    vocab_pad_multiple: int = vocab_pad_multiple

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")


def resolve_model_config(tokenizer_vocab_size: int | None = None) -> ModelConfig:
    """Build model config from globals, padding vocab to nearest multiple of 64."""
    raw_vocab = tokenizer_vocab_size if tokenizer_vocab_size is not None else vocab_size
    padded_vocab = max(
        pad_vocab_size(raw_vocab),
        pad_vocab_size(vocab_size),
    )
    return ModelConfig(
        vocab_size=padded_vocab,
        d_model=n_embd,
        num_heads=n_head,
        num_layers=n_layer,
        max_seq_len=max(max_seq_len, block_size),
        rope_theta=rope_theta,
        rms_eps=rms_eps,
        dropout=dropout,
        bias=bias,
        vocab_pad_multiple=vocab_pad_multiple,
    )


def ensure_out_dir() -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
