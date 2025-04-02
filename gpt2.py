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


class CausalSelfAttention(torch.Module):
    def __init__(self, config: GPTConfig):
        assert config.n_embd % config.n_head == 0
        self.atten = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.atten(x).split(3, dim=-1)
        q = q * (C ** -0.5)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


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