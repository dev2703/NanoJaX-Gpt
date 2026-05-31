#!/usr/bin/env python3
"""
Train a GPT on tokenized text. Hyperparameters live in config.py.

Usage:
  python train_tokenizer.py   # once
  python prepare_data.py      # once
  python train.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import config
from bpe import BPETokenizer
from checkpoint import save_checkpoint
from config import resolve_model_config
from data import TokenDataset, get_batch
from losses import cross_entropy
from model import GPT
from optimizers import create_optimizer
from schedules import cosine_warmup_schedule, get_lr


def main() -> None:
    config.ensure_out_dir()

    if not Path(config.tokens_path).is_file():
        raise SystemExit(
            f"Token cache not found at {config.tokens_path}. "
            "Run: python prepare_data.py"
        )

    tokenizer = BPETokenizer.load(config.tokenizer_path)
    model_cfg = resolve_model_config(len(tokenizer.vocab))
    dataset = TokenDataset(config.tokens_path, val_fraction=config.val_fraction)
    model = GPT(model_cfg)

    key = jax.random.key(config.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, config.block_size), dtype=jnp.int32)
    params = model.init(init_key, dummy)

    schedule = cosine_warmup_schedule(
        base_lr=config.learning_rate,
        warmup=config.warmup_iters,
        max_iters=config.max_iters,
    )
    optimizer = create_optimizer(
        learning_rate=schedule,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.adam_eps,
    )
    opt_state = optimizer.init(params)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(config.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("config.py", run_dir / "config.py")

    @jax.jit
    def train_step(
        params: nn.FrozenDict,
        opt_state: optax.OptState,
        x: jnp.ndarray,
        y: jnp.ndarray,
        step: jnp.ndarray,
        dropout_rng: jax.Array,
    ):
        def loss_fn(p):
            logits = model.apply(p, x, rngs={"dropout": dropout_rng})
            return cross_entropy(logits, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params, step=step)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params: nn.FrozenDict, x: jnp.ndarray, y: jnp.ndarray):
        logits = model.apply(params, x, deterministic=True)
        return cross_entropy(logits, y)

    step = 0
    best_val = float("inf")
    metrics: dict[str, list] = {
        "train_steps": [],
        "train_loss": [],
        "train_loss_raw": [],
        "val_steps": [],
        "val_loss": [],
    }
    loss_ema: float | None = None

    def _save_metrics() -> None:
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

    print(f"Training run: {run_dir}", flush=True)
    print(f"Train tokens: {dataset.train_size:,} | Val tokens: {dataset.val_size:,}", flush=True)
    print(f"Model params: {sum(x.size for x in jax.tree.leaves(params)):,}", flush=True)

    def _eval_val() -> float:
        val_losses: list[float] = []
        nonlocal key
        for _ in range(config.eval_iters):
            key, eval_key = jax.random.split(key)
            vx, vy = get_batch(
                eval_key,
                dataset.val_data,
                config.block_size,
                config.batch_size,
            )
            val_losses.append(float(eval_step(params, vx, vy)))
        return sum(val_losses) / len(val_losses)

    val_loss = _eval_val()
    metrics["val_steps"].append(0)
    metrics["val_loss"].append(val_loss)
    best_val = val_loss
    print(f"step {0:5d} | val_loss {val_loss:.4f}", flush=True)

    while step < config.max_iters:
        key, batch_key, dropout_key = jax.random.split(key, 3)
        x, y = get_batch(batch_key, dataset.train_data, config.block_size, config.batch_size)
        step_jax = jnp.asarray(step, dtype=jnp.int32)
        params, opt_state, loss = train_step(
            params, opt_state, x, y, step_jax, dropout_key
        )
        step += 1

        batch_loss = float(loss)
        if loss_ema is None:
            loss_ema = batch_loss
        else:
            loss_ema = config.ema_decay * loss_ema + (1.0 - config.ema_decay) * batch_loss

        if step % config.log_interval == 0 or step == 1:
            lr = get_lr(schedule, step)
            metrics["train_steps"].append(step)
            metrics["train_loss"].append(loss_ema)
            metrics["train_loss_raw"].append(batch_loss)
            print(
                f"step {step:5d} | loss {loss_ema:.4f} "
                f"(batch {batch_loss:.4f}) | lr {lr:.2e}",
                flush=True,
            )

        if step % config.eval_interval == 0 or step == config.max_iters:
            val_loss = _eval_val()
            metrics["val_steps"].append(step)
            metrics["val_loss"].append(val_loss)
            _save_metrics()
            print(f"step {step:5d} | val_loss {val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    run_dir / "best",
                    params,
                    opt_state,
                    step,
                    metadata={"val_loss": val_loss, "model": "GPT"},
                )

    save_checkpoint(
        run_dir / "last",
        params,
        opt_state,
        step,
        metadata={"val_loss": best_val, "model": "GPT"},
    )
    _save_metrics()
    print(f"Finished training. Checkpoints saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
