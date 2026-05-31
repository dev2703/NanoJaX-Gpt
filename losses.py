from __future__ import annotations

import jax.numpy as jnp
import optax


def cross_entropy(
    logits: jnp.ndarray,
    target: jnp.ndarray,
    reduction: str = "mean",
) -> jnp.ndarray:
    target = target.astype(jnp.int32)
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
        target = target.reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, target)
    if reduction == "mean":
        return jnp.mean(loss)
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction!r}")
