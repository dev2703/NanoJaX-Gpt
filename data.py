from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bpe import BPETokenizer


class TokenDataset:
    def __init__(self, tokens_path: str | Path, val_fraction: float = 0.1) -> None:
        tokens_path = Path(tokens_path)
        if not tokens_path.is_file():
            raise FileNotFoundError(f"Token file not found: {tokens_path}")

        self.data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        if self.data.shape[0] < 2:
            raise ValueError("Token file must contain at least 2 tokens.")

        split_idx = int(len(self.data) * (1.0 - val_fraction))
        split_idx = max(split_idx, 1)
        split_idx = min(split_idx, len(self.data) - 1)

        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

    @property
    def train_size(self) -> int:
        return int(self.train_data.shape[0])

    @property
    def val_size(self) -> int:
        return int(self.val_data.shape[0])


def get_batch(
    key: jax.Array,
    split_data: np.memmap,
    block_size: int,
    batch_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    data_len = int(split_data.shape[0])
    if data_len <= block_size:
        raise ValueError("Not enough tokens for the requested block_size.")

    key, subkey = jax.random.split(key)
    starts = jax.random.randint(subkey, (batch_size,), 0, data_len - block_size - 1)

    x_rows = []
    y_rows = []
    for start in np.asarray(starts):
        chunk = split_data[start : start + block_size + 1]
        x_rows.append(chunk[:-1])
        y_rows.append(chunk[1:])

    x = jnp.asarray(np.stack(x_rows), dtype=jnp.int32)
    y = jnp.asarray(np.stack(y_rows), dtype=jnp.int32)
    return x, y


def encode_corpus_to_memmap(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> int:
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.load(tokenizer_path)
    text = corpus_path.read_text(encoding="utf-8")
    token_ids = np.asarray(tokenizer.encode_text(text), dtype=np.uint16)

    if int(token_ids.max()) >= np.iinfo(np.uint16).max:
        raise ValueError("vocab_size exceeds uint16; use a smaller vocab or switch dtype.")

    memmap = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=token_ids.shape)
    memmap[:] = token_ids
    memmap.flush()
    return int(token_ids.shape[0])
