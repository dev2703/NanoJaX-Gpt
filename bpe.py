from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class BPETrainer:
    DEFAULT_BASE_VOCAB_SIZE = 256

    def __init__(
        self,
        vocab_size: int,
        special_tokens: Sequence[str] | None = None,
        base_vocab_size: int = DEFAULT_BASE_VOCAB_SIZE,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if base_vocab_size <= 0:
            raise ValueError("base_vocab_size must be positive.")
        if base_vocab_size > 256:
            raise ValueError("base_vocab_size must be <= 256 for byte vocabularies.")
        self.vocab_size = vocab_size
        self.base_vocab_size = base_vocab_size
        self.special_tokens = list(special_tokens) if special_tokens else []
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self._validate_vocab_size()

    def _validate_vocab_size(self) -> None:
        min_size = self.base_vocab_size + len(self.special_tokens)
        if self.vocab_size < min_size:
            raise ValueError(
                f"vocab_size must be >= {min_size} (base {self.base_vocab_size} + specials)."
            )

    def _init_vocab(self) -> int:
        self.vocab = {idx: bytes([idx]) for idx in range(self.base_vocab_size)}
        current_id = self.base_vocab_size
        for token_str in self.special_tokens:
            self.vocab[current_id] = token_str.encode("utf-8")
            current_id += 1
        return current_id

    @staticmethod
    def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(ids: list[int], new_token_id: int, pair: tuple[int, int]) -> list[int]:
        new_ids: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_token_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _train_on_ids(self, ids: list[int]) -> None:
        self.merges = []
        next_merge_id = self._init_vocab()
        num_merges = self.vocab_size - self.base_vocab_size - len(self.special_tokens)

        if num_merges <= 0:
            return

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break

            pair, count = max(stats.items(), key=lambda item: item[1])
            new_token_id = next_merge_id + i
            ids = self.merge(ids, new_token_id, pair)

            byte_pair_0 = self.vocab[pair[0]]
            byte_pair_1 = self.vocab[pair[1]]
            self.vocab[new_token_id] = byte_pair_0 + byte_pair_1
            self.merges.append((byte_pair_0, byte_pair_1))

    def train(self, data: bytes | bytearray) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes or bytearray.")
        data_bytes = bytes(data)
        if data_bytes:
            max_byte = max(data_bytes)
            if max_byte >= self.base_vocab_size:
                raise ValueError(
                    "data contains byte values outside base_vocab_size; "
                    "increase base_vocab_size or preprocess the data."
                )
        self._train_on_ids(list(data_bytes))
        return self.vocab, self.merges

    def train_from_path(self, path: str | Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        return self.train(file_path.read_bytes())


@dataclass(frozen=True)
class BPETokenizerConfig:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    base_vocab_size: int = 256
    encoding: str = "utf-8"
    special_tokens: Sequence[str] | None = None


class BPETokenizer:
    def __init__(self, config: BPETokenizerConfig):
        self.vocab = dict(config.vocab)
        self.merges = list(config.merges)
        self.base_vocab_size = config.base_vocab_size
        self.encoding = config.encoding
        self.special_tokens = list(config.special_tokens) if config.special_tokens else []
        self.token_to_id = self._build_token_to_id(self.vocab)
        self.pair_ranks = {pair: idx for idx, pair in enumerate(self.merges)}
        self._validate_vocab()

    @staticmethod
    def _build_token_to_id(vocab: dict[int, bytes]) -> dict[bytes, int]:
        token_to_id: dict[bytes, int] = {}
        for idx in sorted(vocab):
            token = vocab[idx]
            if token not in token_to_id:
                token_to_id[token] = idx
        return token_to_id

    @classmethod
    def from_trainer(cls, trainer: BPETrainer, encoding: str = "utf-8") -> BPETokenizer:
        return cls(
            BPETokenizerConfig(
                vocab=trainer.vocab,
                merges=trainer.merges,
                base_vocab_size=trainer.base_vocab_size,
                encoding=encoding,
                special_tokens=trainer.special_tokens,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        file_path = Path(path)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        vocab = {int(token_id): _decode_bytes(token) for token_id, token in payload["vocab"].items()}
        merges = [(_decode_bytes(left), _decode_bytes(right)) for left, right in payload["merges"]]
        return cls(
            BPETokenizerConfig(
                vocab=vocab,
                merges=merges,
                base_vocab_size=payload.get("base_vocab_size", 256),
                encoding=payload.get("encoding", "utf-8"),
                special_tokens=payload.get("special_tokens"),
            )
        )

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        payload = {
            "base_vocab_size": self.base_vocab_size,
            "encoding": self.encoding,
            "special_tokens": self.special_tokens,
            "vocab": {str(token_id): _encode_bytes(token) for token_id, token in self.vocab.items()},
            "merges": [[_encode_bytes(left), _encode_bytes(right)] for left, right in self.merges],
        }
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _validate_vocab(self) -> None:
        if self.base_vocab_size <= 0:
            raise ValueError("base_vocab_size must be positive.")
        if self.base_vocab_size > 256:
            raise ValueError("base_vocab_size must be <= 256 for byte vocabularies.")
        for idx in range(self.base_vocab_size):
            if idx not in self.vocab:
                raise ValueError("vocab missing base byte token ids.")
            if self.vocab[idx] != bytes([idx]):
                raise ValueError("base byte tokens must map to single-byte values.")
        for token in self.special_tokens:
            token_bytes = token.encode(self.encoding)
            if token_bytes not in self.token_to_id:
                raise ValueError("special token missing from vocab.")
        for left, right in self.merges:
            merged = left + right
            if merged not in self.token_to_id:
                raise ValueError(f"merge result not present in vocab: {merged!r}")

    def encode_text(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_bytes(text.encode(self.encoding, errors="surrogatepass"))

        segments = self._split_text_by_special_tokens(text)
        token_ids: list[int] = []
        for segment in segments:
            if segment in self.special_tokens:
                token_bytes = segment.encode(self.encoding)
                token_ids.append(self.token_to_id[token_bytes])
            else:
                token_ids.extend(
                    self.encode_bytes(segment.encode(self.encoding, errors="surrogatepass"))
                )
        return token_ids

    def encode_bytes(self, data: bytes) -> list[int]:
        if not data:
            return []
        if max(data) >= self.base_vocab_size:
            raise ValueError(
                "data contains byte values outside base_vocab_size; "
                "increase base_vocab_size or preprocess the data."
            )
        tokens = [bytes([value]) for value in data]
        tokens = self._bpe_merge(tokens)
        token_ids: list[int] = []
        for token in tokens:
            token_id = self.token_to_id.get(token)
            if token_id is None:
                raise ValueError(f"token not present in vocab: {token!r}")
            token_ids.append(token_id)
        return token_ids

    def decode_ids(self, ids: Sequence[int]) -> bytes:
        chunks: list[bytes] = []
        for token_id in ids:
            token = self.vocab.get(int(token_id))
            if token is None:
                raise ValueError(f"token id not present in vocab: {token_id}")
            chunks.append(token)
        return b"".join(chunks)

    def decode_text(self, ids: Sequence[int], errors: str = "replace") -> str:
        return self.decode_ids(ids).decode(self.encoding, errors=errors)

    def _bpe_merge(self, tokens: list[bytes]) -> list[bytes]:
        if not self.pair_ranks or len(tokens) < 2:
            return tokens

        while True:
            pairs = {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}
            if not pairs:
                break
            best_pair = min(pairs, key=lambda pair: self.pair_ranks.get(pair, float("inf")))
            if best_pair not in self.pair_ranks:
                break
            tokens = self._merge_pair(tokens, best_pair)
        return tokens

    @staticmethod
    def _merge_pair(tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        merged: list[bytes] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _split_text_by_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        ordered_tokens = sorted(self.special_tokens, key=len, reverse=True)
        parts: list[str] = []
        buffer: list[str] = []
        i = 0
        while i < len(text):
            matched = None
            for token in ordered_tokens:
                if text.startswith(token, i):
                    matched = token
                    break
            if matched is not None:
                if buffer:
                    parts.append("".join(buffer))
                    buffer = []
                parts.append(matched)
                i += len(matched)
            else:
                buffer.append(text[i])
                i += 1
        if buffer:
            parts.append("".join(buffer))
        return parts


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_bytes(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))
