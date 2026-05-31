from __future__ import annotations

from typing import Any

import optax


def create_optimizer(
    learning_rate: float | optax.Schedule,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )
    if grad_clip > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optimizer,
        )
    return optimizer


def apply_optimizer_step(
    params: Any,
    grads: Any,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[Any, optax.OptState]:
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
