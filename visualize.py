from __future__ import annotations

import json
from pathlib import Path


def load_metrics(metrics_path: str | Path) -> dict:
    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 2:
        return list(values)
    window = min(window, len(values))
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _truncate_to_step(steps: list, values: list, max_step: int | None) -> tuple[list, list]:
    if max_step is None:
        return steps, values
    out_steps, out_values = [], []
    for s, v in zip(steps, values):
        if s <= max_step:
            out_steps.append(s)
            out_values.append(v)
    return out_steps, out_values


def plot_training_curves(
    metrics_path: str | Path,
    output_path: str | Path,
    title: str = "NanoJaX-Gpt training loss",
    smooth_window: int = 1,
    show_raw_train: bool = False,
    val_only: bool = False,
    max_step: int | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = load_metrics(metrics_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    val_steps = metrics.get("val_steps", [])
    val_loss = metrics.get("val_loss", [])
    val_steps, val_loss = _truncate_to_step(val_steps, val_loss, max_step)

    if val_only:
        if not val_steps or not val_loss:
            raise ValueError("metrics.json has no val_steps / val_loss entries.")
        if smooth_window > 1:
            val_loss = _moving_average(val_loss, smooth_window)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(val_steps, val_loss, label="val/loss", color="#2563eb", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    fig, ax = plt.subplots(figsize=(10, 5))

    train_steps = metrics.get("train_steps", [])
    train_loss = metrics.get("train_loss", [])
    train_raw = metrics.get("train_loss_raw", [])
    train_steps, train_loss = _truncate_to_step(train_steps, train_loss, max_step)
    _, train_raw = _truncate_to_step(metrics.get("train_steps", []), train_raw, max_step)

    if train_steps and train_loss:
        plot_train = (
            _moving_average(train_loss, smooth_window) if smooth_window > 1 else train_loss
        )
        ax.plot(
            train_steps,
            plot_train,
            label="train loss (EMA)",
            color="#2563eb",
            linewidth=2,
        )
        if show_raw_train and train_raw:
            ax.plot(
                train_steps,
                train_raw,
                label="train loss (batch)",
                color="#93c5fd",
                alpha=0.45,
                linewidth=1,
            )

    if val_steps and val_loss:
        ax.plot(
            val_steps,
            val_loss,
            label="val loss",
            color="#dc2626",
            marker="o",
            linewidth=2,
            markersize=4,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
