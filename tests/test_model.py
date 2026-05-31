from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("flax")

from config import ModelConfig
from model import GPT


def test_gpt_forward_shape() -> None:
    cfg = ModelConfig(
        vocab_size=128,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=32,
    )
    model = GPT(cfg)
    key = jax.random.key(0)
    token_ids = jax.random.randint(key, (2, 16), 0, cfg.vocab_size)
    params = model.init(key, token_ids)
    logits = model.apply(params, token_ids)
    assert logits.shape == (2, 16, cfg.vocab_size)
