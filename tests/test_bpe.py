from __future__ import annotations

import tempfile
from pathlib import Path

from bpe import BPETrainer, BPETokenizer


def test_bpe_round_trip_and_save_load() -> None:
    trainer = BPETrainer(vocab_size=300, special_tokens=["<|endoftext|>"])
    trainer.train(b"hello world " * 40)
    tokenizer = BPETokenizer.from_trainer(trainer)

    text = "hello world"
    ids = tokenizer.encode_text(text)
    assert tokenizer.decode_text(ids) == text

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokenizer.json"
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.decode_text(loaded.encode_text(text)) == text
