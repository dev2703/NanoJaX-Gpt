"""
Fast training preset — J-shaped val/loss curve in ~10–20 min on Mac CPU.

Initial loss ≈ ln(512) ≈ 6.2 (random init). With enough steps a 6×384 model
on Tiny Shakespeare typically reaches val loss ~2–2.5.

Usage:
  python train_fast.py
  python plot_metrics.py --metrics out/fast/run_.../metrics.json --val-only \\
      --output docs/images/val_loss_fast.png
"""

from __future__ import annotations

import config

# Separate output dir so fast runs don't overwrite full training.
config.out_dir = "out/fast"
config.tokenizer_path = f"{config.out_dir}/tokenizer.json"
config.tokens_path = f"{config.out_dir}/tokens.bin"

# Data
config.block_size = 256
config.batch_size = 32
config.val_fraction = 0.1

# Model — ~8M params (vs ~91M for 12×792); still enough to reach val ~2
config.vocab_size = 512
config.n_layer = 6
config.n_head = 6
config.n_embd = 384
config.dropout = 0.1
config.bias = True
config.max_seq_len = 256
config.rms_eps = 1e-5

# Optimizer
config.learning_rate = 3e-4
config.max_iters = 2000
config.warmup_iters = 100
config.weight_decay = 0.1
config.grad_clip = 1.0
config.beta1 = 0.9
config.beta2 = 0.95

# Logging — eval every 50 steps → ~40 val points on the curve
config.eval_interval = 50
config.log_interval = 20
config.eval_iters = 5
config.seed = 42
config.ema_decay = 0.98
