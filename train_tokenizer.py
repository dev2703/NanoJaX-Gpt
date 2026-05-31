#!/usr/bin/env python3
"""Train byte-level BPE tokenizer. Settings in config.py."""
from __future__ import annotations

from pathlib import Path

import config
from bpe import BPETrainer, BPETokenizer


def main() -> None:
    config.ensure_out_dir()
    output_path = Path(config.tokenizer_path)

    trainer = BPETrainer(
        vocab_size=config.vocab_size,
        special_tokens=config.special_tokens,
    )
    trainer.train_from_path(config.dataset)
    tokenizer = BPETokenizer.from_trainer(trainer)
    tokenizer.save(output_path)

    print(f"Trained on: {Path(config.dataset).resolve()}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Saved tokenizer to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
