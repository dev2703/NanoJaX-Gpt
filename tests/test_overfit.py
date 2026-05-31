from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("flax")
pytest.importorskip("optax")

from config import ModelConfig
from data import get_batch
from losses import cross_entropy
from model import GPT
from optimizers import apply_optimizer_step, create_optimizer


def test_overfit_single_batch() -> None:
    vocab_size = 64
    block_size = 16
    batch_size = 2

    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=block_size,
    )
    model = GPT(cfg)

    rng = np.random.default_rng(0)
    data = np.asarray(rng.integers(0, vocab_size, size=512), dtype=np.uint16)

    key = jax.random.key(0)
    x, y = get_batch(key, data, block_size, batch_size)
    params = model.init(key, x)

    optimizer = create_optimizer(learning_rate=1e-2, weight_decay=0.0, grad_clip=0.0)
    opt_state = optimizer.init(params)

    initial_loss = float(cross_entropy(model.apply(params, x), y))

    for step in range(50):
        def loss_fn(p):
            return cross_entropy(model.apply(p, x), y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, opt_state = apply_optimizer_step(params, grads, optimizer, opt_state)

    final_loss = float(loss)
    assert final_loss < initial_loss * 0.5
