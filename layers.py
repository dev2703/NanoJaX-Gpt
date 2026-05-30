from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class LinearLayer(nn.Module):
    in_features: int
    out_features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param(
            "weight",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.in_features, self.out_features),
            self.dtype,
        )
        return x @ weight


class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param("weight", nn.initializers.ones, (self.d_model,), self.dtype)
        original_dtype = x.dtype
        x_float = x.astype(jnp.float32)
        mean_square = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
        x_normed = x_float * jax.lax.rsqrt(mean_square + self.eps)
        return (weight * x_normed).astype(original_dtype)


class SwiGLU(nn.Module):
    d_model: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        d_ff = int(2 * (4 * self.d_model / 3))
        d_ff = 64 * ((d_ff + 64 - 1) // 64)
        self.w = nn.Dense(d_ff, use_bias=False, dtype=self.dtype)
        self.v = nn.Dense(d_ff, use_bias=False, dtype=self.dtype)
        self.u = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.silu(self.w(x))
        return self.u(gate * self.v(x))


class RotaryPositionalEmbedding(nn.Module):
    theta: float
    d_k: int
    max_seq_len: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        if self.d_k % 2 != 0:
            raise ValueError("d_k must be even for rotary embeddings.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        inv_freq = 1.0 / (
            self.theta ** (jnp.arange(0, self.d_k, 2, dtype=jnp.float32) / self.d_k)
        )
        positions = jnp.arange(self.max_seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", positions, inv_freq)
        self.variable(
            "cache",
            "cos",
            lambda: freqs.cos().astype(self.dtype),
            mutable=False,
        )
        self.variable(
            "cache",
            "sin",
            lambda: freqs.sin().astype(self.dtype),
            mutable=False,
        )

    def __call__(self, x: jnp.ndarray, token_positions: jnp.ndarray) -> jnp.ndarray:
        if token_positions.size == 0:
            return x

        token_positions = token_positions.astype(jnp.int32)
        if jnp.max(token_positions) >= self.max_seq_len:
            raise ValueError("token_positions contains values >= max_seq_len")
        if x.shape[-1] != self.d_k:
            raise ValueError("x last dimension must match d_k")

        cos_table = self.get_variable("cache", "cos")
        sin_table = self.get_variable("cache", "sin")

        x_reshaped = x.reshape(*x.shape[:-1], self.d_k // 2, 2)

        if token_positions.ndim == 1:
            cos = cos_table[token_positions][None, :, None, :]
            sin = sin_table[token_positions][None, :, None, :]
        elif token_positions.ndim == 2:
            cos = cos_table[token_positions][..., None, :]
            sin = sin_table[token_positions][..., None, :]
        else:
            raise ValueError("token_positions must be 1D or 2D")

        cos = cos.astype(x.dtype)[..., None]
        sin = sin.astype(x.dtype)[..., None]

        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        rotated = jnp.stack((-x2, x1), axis=-1)
        out = (x_reshaped * cos) + (rotated * sin)
        return out.reshape(x.shape)
