from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from config import ModelConfig


class LinearLayer(nn.Module):
    in_features: int
    out_features: int
    dtype: jnp.dtype = jnp.float32
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(
            self.out_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
        )(x)


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
    use_bias: bool = False
    dropout: float = 0.0

    def setup(self) -> None:
        d_ff = int(2 * (4 * self.d_model / 3))
        d_ff = 64 * ((d_ff + 64 - 1) // 64)
        self.w = nn.Dense(d_ff, use_bias=self.use_bias, dtype=self.dtype)
        self.v = nn.Dense(d_ff, use_bias=self.use_bias, dtype=self.dtype)
        self.u = nn.Dense(self.d_model, use_bias=self.use_bias, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(self.dropout)

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        gate = nn.silu(self.w(x))
        return self.dropout_layer(self.u(gate * self.v(x)), deterministic=deterministic)


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
        cos_table = jnp.cos(freqs).astype(self.dtype)
        sin_table = jnp.sin(freqs).astype(self.dtype)
        self.variable("cache", "cos", lambda *_, **__: cos_table)
        self.variable("cache", "sin", lambda *_, **__: sin_table)

    def __call__(self, x: jnp.ndarray, token_positions: jnp.ndarray) -> jnp.ndarray:
        token_positions = token_positions.astype(jnp.int32)

        cos_table = self.get_variable("cache", "cos")
        sin_table = self.get_variable("cache", "sin")

        x_reshaped = x.reshape(*x.shape[:-1], self.d_k // 2, 2)

        if token_positions.ndim == 1:
            cos = cos_table[token_positions]
            sin = sin_table[token_positions]
            if x.ndim == 4:
                cos = cos[None, None, :, :, None]
                sin = sin[None, None, :, :, None]
            else:
                cos = cos[None, :, :, None]
                sin = sin[None, :, :, None]
        elif token_positions.ndim == 2:
            cos = cos_table[token_positions][:, None, :, :, None]
            sin = sin_table[token_positions][:, None, :, :, None]
        else:
            raise ValueError("token_positions must be 1D or 2D")

        cos = cos.astype(x.dtype)
        sin = sin.astype(x.dtype)

        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        rotated = jnp.stack((-x2, x1), axis=-1)
        out = (x_reshaped * cos) + (rotated * sin)
        return out.reshape(x.shape)


def scaled_dot_product_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask, attn_logits, -1e9)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


def expand_mask(mask: jnp.ndarray) -> jnp.ndarray:
    if mask.ndim < 2:
        raise ValueError("Mask must be at least 2-dimensional (seq_len, seq_len).")
    if mask.ndim == 3:
        mask = jnp.expand_dims(mask, 1)
    while mask.ndim < 4:
        mask = jnp.expand_dims(mask, 0)
    return mask


class MultiHeadSelfAttention(nn.Module):
    d_model: int
    num_heads: int
    dtype: jnp.dtype = jnp.float32
    use_bias: bool = False
    dropout: float = 0.0

    def setup(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.head_dim = self.d_model // self.num_heads
        self.q_proj = nn.Dense(self.d_model, use_bias=self.use_bias, dtype=self.dtype)
        self.k_proj = nn.Dense(self.d_model, use_bias=self.use_bias, dtype=self.dtype)
        self.v_proj = nn.Dense(self.d_model, use_bias=self.use_bias, dtype=self.dtype)
        self.o_proj = nn.Dense(self.d_model, use_bias=self.use_bias, dtype=self.dtype)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def __call__(
        self,
        x: jnp.ndarray,
        rope: RotaryPositionalEmbedding,
        positions: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        batch_size, seq_length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = rope(q, positions)
        k = rope(k, positions)

        if mask is not None:
            mask = expand_mask(mask)

        values, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        values = self.attn_dropout(values, deterministic=deterministic)
        values = values.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.d_model)
        return self.resid_dropout(self.o_proj(values), deterministic=deterministic)


class TransformerBlock(nn.Module):
    config: ModelConfig

    def setup(self) -> None:
        self.norm1 = RMSNorm(
            self.config.d_model,
            eps=self.config.rms_eps,
            dtype=self.config.dtype,
        )
        self.attn = MultiHeadSelfAttention(
            self.config.d_model,
            self.config.num_heads,
            dtype=self.config.dtype,
            use_bias=self.config.bias,
            dropout=self.config.dropout,
        )
        self.norm2 = RMSNorm(
            self.config.d_model,
            eps=self.config.rms_eps,
            dtype=self.config.dtype,
        )
        self.ffn = SwiGLU(
            self.config.d_model,
            dtype=self.config.dtype,
            use_bias=self.config.bias,
            dropout=self.config.dropout,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        rope: RotaryPositionalEmbedding,
        positions: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        x = x + self.attn(self.norm1(x), rope, positions, mask, deterministic=deterministic)
        x = x + self.ffn(self.norm2(x), deterministic=deterministic)
        return x


class GPT(nn.Module):
    """Decoder-only Transformer language model."""

    config: ModelConfig

    def setup(self) -> None:
        self.embed = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            dtype=self.config.dtype,
        )
        self.rope = RotaryPositionalEmbedding(
            theta=self.config.rope_theta,
            d_k=self.config.d_model // self.config.num_heads,
            max_seq_len=self.config.max_seq_len,
            dtype=self.config.dtype,
        )
        self.blocks = [
            TransformerBlock(self.config) for _ in range(self.config.num_layers)
        ]
        self.final_norm = RMSNorm(
            self.config.d_model,
            eps=self.config.rms_eps,
            dtype=self.config.dtype,
        )
        self.output_head = LinearLayer(
            self.config.d_model,
            self.config.vocab_size,
            dtype=self.config.dtype,
            use_bias=self.config.bias,
        )

    def __call__(
        self,
        token_ids: jnp.ndarray,
        token_positions: jnp.ndarray | None = None,
        deterministic: bool = False,
    ) -> jnp.ndarray:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be 2D (batch, seq_len)")

        _, seq_len = token_ids.shape

        if token_positions is None:
            token_positions = jnp.arange(seq_len, dtype=jnp.int32)
        if token_positions.ndim == 1 and token_positions.shape[0] != seq_len:
            raise ValueError("token_positions length must match sequence length")
        if token_positions.ndim == 2 and token_positions.shape[1] != seq_len:
            raise ValueError("token_positions shape must match sequence length")

        x = self.embed(token_ids)
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

        for block in self.blocks:
            x = block(x, self.rope, token_positions, mask, deterministic=deterministic)

        return self.output_head(self.final_norm(x))
