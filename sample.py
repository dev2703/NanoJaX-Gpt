#!/usr/bin/env python3
"""Sample text from a trained checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from bpe import BPETokenizer
from checkpoint import load_checkpoint
from config import resolve_model_config
from model import GPT


def _sample_next_token(
    key: jax.Array,
    logits: jnp.ndarray,
    temperature: float,
    top_k: int | None,
) -> jnp.ndarray:
    logits = logits.astype(jnp.float32) / max(temperature, 1e-8)
    if top_k is not None and top_k > 0:
        top_values = jnp.sort(logits)[-top_k:]
        cutoff = top_values[0]
        logits = jnp.where(logits < cutoff, -1e9, logits)
    probs = jax.nn.softmax(logits, axis=-1)
    return jax.random.categorical(key, probs)


def generate(
    checkpoint_dir: str | Path,
    prompt: str,
    tokenizer_path: str | Path,
    max_new_tokens: int = 100,
    block_size: int = 1024,
    temperature: float = 1.0,
    top_k: int | None = 40,
    seed: int = 0,
) -> str:
    checkpoint_dir = Path(checkpoint_dir)
    params, _, _, _ = load_checkpoint(checkpoint_dir)

    tokenizer = BPETokenizer.load(tokenizer_path)
    model_cfg = resolve_model_config(len(tokenizer.vocab))
    model = GPT(model_cfg)

    prompt_ids = jnp.asarray([tokenizer.encode_text(prompt)], dtype=jnp.int32)
    key = jax.random.key(seed)
    generated = prompt_ids

    for _ in range(max_new_tokens):
        context = generated[:, -block_size:]
        logits = model.apply(params, context, deterministic=True)
        next_logits = logits[:, -1, :]
        key, subkey = jax.random.split(key)
        next_id = _sample_next_token(subkey, next_logits[0], temperature, top_k)
        generated = jnp.concatenate([generated, next_id[None, None]], axis=1)

    all_ids = [int(token_id) for token_id in generated[0].tolist()]
    return tokenizer.decode_text(all_ids)


def main() -> None:
    import config as cfg

    parser = argparse.ArgumentParser(description="Sample text from a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--tokenizer", type=Path, default=Path(cfg.tokenizer_path))
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=cfg.block_size)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    text = generate(
        checkpoint_dir=args.checkpoint,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
