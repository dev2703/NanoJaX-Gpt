#!/usr/bin/env python3
"""Plot train/val loss from metrics.json."""
from __future__ import annotations

import argparse
from pathlib import Path

from visualize import plot_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/val loss from metrics.json.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/images/training_loss.png"))
    parser.add_argument("--title", type=str, default="NanoJaX-Gpt training loss")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--show-raw-train", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only plot metrics up to this training step (inclusive).",
    )
    args = parser.parse_args()

    out = plot_training_curves(
        args.metrics,
        args.output,
        title=args.title,
        smooth_window=args.smooth_window,
        show_raw_train=args.show_raw_train,
        val_only=args.val_only,
        max_step=args.max_step,
    )
    print(f"Saved plot to {out.resolve()}")


if __name__ == "__main__":
    main()
