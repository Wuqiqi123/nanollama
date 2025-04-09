import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import einx.nn.torch as einn
import einx
import math

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
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


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        ## parameter share 
        # TODO(wqq): the shape of wte are transpose of lm_head, Does it right?
        self.transformer.wte.weight = self.lm_head.weight


        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))


    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long) # (T)

        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.layers:
            x = block(x)
        x = self.transformer.ln_f(x) # (B, T, n_embd)

        if targets is not None:
            logits = self.lm_head(x)  # => (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss


if __name__ == "__main__":
    config = GPTConfig()
    gpt = GPT(config)
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    print(f"x.shape: {x.shape}")
    target = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    y = gpt(x, target)
    print(f"y.shape: {y[0].shape}")



