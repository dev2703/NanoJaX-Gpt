from __future__ import annotations

import optax


def cosine_warmup_schedule(
    base_lr: float,
    warmup: int,
    max_iters: int,
    end_lr: float = 0.0,
) -> optax.Schedule:
    if warmup <= 0 or max_iters <= 0:
        raise ValueError("warmup and max_iters must be positive.")
    if warmup >= max_iters:
        raise ValueError("warmup must be less than max_iters.")

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup,
        decay_steps=max_iters - warmup,
        end_value=end_lr,
    )


def get_lr(schedule: optax.Schedule, step: int) -> float:
    return float(schedule(step))
