# NanoJaX-Gpt

A from-scratch GPT-style language model in **JAX / Flax / Optax**, structured like [NanoGPT](https://github.com/karpathy/nanoGPT): a flat repo with a few Python files at the root, hyperparameters in `config.py`, and scripts you run directly. Inspired by Karpathy's NanoGPT and Stanford CS336 Assignment 1.

Byte-level BPE tokenization, pre-norm Transformer blocks (RMSNorm, RoPE, SwiGLU), AdamW training with cosine warmup, and autoregressive sampling.
<img width="1500" height="750" alt="val_loss_fast_1000" src="https://github.com/user-attachments/assets/9bef4b61-26f2-446c-828d-5e5636247f9c" />


---

## Project structure

Like NanoGPT — everything lives at the repo root:

| Package | Purpose |
|---------|---------|
| `jax`, `jaxlib` | Array ops, `jit`, autodiff, devices (CPU/GPU/TPU) |
| `flax` | `nn.Module`, parameters, checkpoint serialization |
| `optax` | AdamW/SGD, LR schedules, gradient clipping |
| `numpy` | Token memmap dataset |
| `pyyaml` | Model/train configs |
| `msgpack` | Checkpoint payloads (via Flax) |
| `matplotlib` | Training loss plots (`scripts/plot_metrics.py`) |
---

## Dependencies

| Package | Purpose |
|---------|---------|
| `jax`, `jaxlib` | Arrays, `jit`, autodiff, CPU/GPU/TPU |
| `flax` | `nn.Module`, checkpoint serialization |
| `optax` | AdamW, LR schedules, grad clipping |
| `numpy` | Token memmap |
| `matplotlib` | Loss plots |

### Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**JAX on GPU:** follow the [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html) for your platform. Apple Silicon usually gets Metal with `pip install jax`.

**Python:** 3.11+

---

## Dataset

Default corpus: `Data/input.txt` (Tiny Shakespeare).

```bash
mkdir -p Data
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o Data/input.txt
```

If you wanna train model on bigger corpus. Here's few more examples.
```bash
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
---

## Quick start

All commands from the **repo root**:

```bash
# 1. Train BPE tokenizer (512 vocab by default)
python train_tokenizer.py

# 2. Encode text to token cache
python prepare_data.py

# 3. Train
python train.py

# 4. Sample
python sample.py --checkpoint out/run_YYYYMMDD_HHMMSS/best --prompt "ROMEO:"

# 5. Plot loss
python plot_metrics.py --metrics out/run_YYYYMMDD_HHMMSS/metrics.json
```

Checkpoints land in `out/run_*/best` and `out/run_*/last`. Each run saves a copy of `config.py` and `metrics.json`.


## Configuration

All hyperparameters are plain Python variables in [`config.py`](config.py). Defaults match NanoGPT Shakespeare training:

```python
block_size = 1024
batch_size = 32
vocab_size = 512
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0025
bias = False
learning_rate = 6e-4
max_iters = 5000
```

Vocab is padded to the nearest multiple of 64 for efficiency. 

## Troubleshooting

**`ModuleNotFoundError: jax`** — run `pip install -r requirements.txt` from repo root.

**Tokenizer / tokens not found** — run `train_tokenizer.py` then `prepare_data.py` before `train.py`.

**BPE training slow** — large `vocab_size` on a big file can take a while. Start with `vocab_size = 512`.

**JAX OOM** — reduce `batch_size`, `n_embd`, `n_layer`, or `block_size` in `config.py`.

**Garbled samples** — train longer; 5000 steps on CPU takes time. Lower temperature (`0.5–0.8`) helps.

**Verify device:**

```python
import jax
print(jax.devices())
```

---
