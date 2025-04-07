import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import einx.nn.torch as einn
import einx
from functools import partial

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = -1
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    use_flash_attn: bool = False
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)
        self.use_flash_attn = config.use_flash_attn
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = einx.rearrange("b q (h c) -> b h q c", q, h=self.n_head)
        k = einx.rearrange("b k (h c) -> b h k c", k, h=self.n_head)
        v = einx.rearrange("b v (h c) -> b h v c", v, h=self.n_head)

        if self.use_flash_attn:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
            attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            einx.dot("b h q [c], b h k [c] -> b h q k", q, k) / ( q.shape[-1] ** 0.5)


    # def __call__(self, x):
    #     # ########### Attention block ###########
    #     x0 = x
    #     x = Norm()(x)

    #     # Predict queries, keys and values
    #     x = Linear(channels=3 * x.shape[-1])(x)
    #     q, k, v = jnp.split(x, 3, axis=-1)

    #     # Compute attention matrix over h heads
    #     q = q * ((q.shape[-1] // self.heads) ** -0.5)
    #     attn = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=self.heads)

    #     # Apply causal mask
    #     mask = jnp.tril(jnp.ones((q.shape[1], q.shape[1]), dtype=bool))
    #     attn = einx.where("q k, b q k h,", mask, attn, -jnp.inf)

    #     # Apply softmax and compute weighted average over the input tokens
    #     attn = einx.softmax("b q [k] h", attn)
    #     x = einx.dot("b q k h, b k (h c) -> b q (h c)", attn, v)

    #     # Output projection
    #     x = Linear(channels=x.shape[-1])(x)

    #     x = x + x0

    #     # ########### MLP block ###########
    #     x0 = x
    #     x = Norm()(x)

    #     x = Linear(channels=x.shape[-1] * self.mlp_ratio)(x)
    #     x = jax.nn.gelu(x)
    #     x = Linear(channels=x0.shape[-1])(x)

    #     x = x + x0

    #     return x


if __name__ == "__main__":
    config = GPTConfig()
    attention = CausalSelfAttention(config)
    x = torch.randn(1, 1024, 768)
    attention(x)

