from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flax import serialization


def save_checkpoint(
    checkpoint_dir: str | Path,
    params: Any,
    opt_state: Any,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "params": params,
        "opt_state": opt_state,
        "step": step,
    }
    (checkpoint_dir / "state.msgpack").write_bytes(serialization.to_bytes(state))

    meta = {"step": step, **(metadata or {})}
    (checkpoint_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return checkpoint_dir


def load_checkpoint(checkpoint_dir: str | Path) -> tuple[Any, Any, int, dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "state.msgpack"
    if not state_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {state_path}")

    state = serialization.from_bytes(None, state_path.read_bytes())
    metadata_path = checkpoint_dir / "metadata.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return state["params"], state["opt_state"], int(state["step"]), metadata
