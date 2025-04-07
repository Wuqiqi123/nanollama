import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import einx.nn.torch as einn
import einx

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = -1
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    use_flash_attn: bool = False
    dropout: float = 0.0
    bias: bool = True


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
        self.output_dropout = nn.Dropout(config.dropout)

        self.register_buffer("causal_mask", torch.ones(1, 1, config.block_size, config.block_size).tril(diagonal=0))


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
            att = einx.dot("b h q [c], b h k [c] -> b h q k", q, k) / ( q.shape[-1] ** 0.5)
            att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = einx.dot("b h q [k], b h [k] c -> b h q c", att, v)
        
        y = einx.rearrange("b h q c -> b q (h c)", y)

        return self.output_dropout(self.o_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == "__main__":
    config = GPTConfig()
    trans = TransformerBlock(config)
    x = torch.randn(1, 1024, 768)
    trans(x)

