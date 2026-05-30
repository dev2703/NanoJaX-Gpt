import jax.numpy as jnp
from flax import linen as nn


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    attn_logits = jnp.matmul(Q, K.transpose(-2, -1)) / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -1e9, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, V)
    return values, attention

def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask
    
class MultiHeadAttention(nn.Module):
    embed_dim: int
    num_heads: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  
                                 bias_init=nn.initializers.zeros  
                                )
        self.o_proj = nn.Dense(self.embed_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)
                               
    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size,seq_length,self.num_heads,self.embed_dim//self.num_heads)
        qkv = qkv.transpose(0,2,1,3)
        q,k,v = jnp.array_split(qkv,3,axis=1)

        values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
        values = values.transpose(0,2,1,3).reshape(batch_size,seq_length,embed_dim)
        output = self.o_proj(values)
        return output,attention

