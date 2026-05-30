from attention import MultiHeadAttention
import jax.numpy as jnp
from flax import linen as nn



class EncoderBlock(nn.Module):
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attention = MultiHeadAttention(
            embed_dim=self.input_dim,
            num_heads=self.num_heads,
            dtype=self.dtype,
        )

        self.feed_forward_layer = nn.Sequential([
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.silu(),
            nn.Dense(self.input_dim),
        ])
        self.norm1 = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.dropout_prob)

    def __call__(self, x, mask=None, deterministic: bool = True):
        attn_output, _ = self.self_attention(x, mask=mask)
        x = x + self.dropout(attn_output, deterministic=deterministic)
        x = self.norm1(x)

        linear_out = self.feed_forward_layer(x, deterministic=deterministic)
        x = x + self.dropout(linear_out, deterministic=deterministic)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    num_layers : int
    input_dim : int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float
    
    def setup(self):
        self.layers = [EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob) for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            x = l(x, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
    
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = l(x, mask=mask, train=train)
        return attention_maps

