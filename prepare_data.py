#!/usr/bin/env python3
"""Encode corpus to token memmap. Settings in config.py."""
from __future__ import annotations

from pathlib import Path

import config
from data import encode_corpus_to_memmap


def main() -> None:
    config.ensure_out_dir()
    tokenizer_path = Path(config.tokenizer_path)
    if not tokenizer_path.is_file():
        raise SystemExit(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run: python train_tokenizer.py"
        )

    num_tokens = encode_corpus_to_memmap(
        config.dataset,
        config.tokenizer_path,
        config.tokens_path,
    )
    print(f"Encoded {num_tokens:,} tokens to {Path(config.tokens_path).resolve()}")


if __name__ == "__main__":
    main()
