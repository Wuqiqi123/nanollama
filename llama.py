import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = -1
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        assert config.n_embd % config.n_head == 0
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_block_size = config.block_size
        self.n_ctx = config.block_size
        self.wq = nn.Parameter(torch.randn((self.n_embd, self.n_embd)))
        self.wk = nn.Parameter(torch.randn((self.n_embd, self.n_embd)))
        self.wv = nn.Parameter(torch.randn((self.n_embd, self.n_embd)))
        self.wo = nn.Parameter(torch.randn((self.n_embd, self.n_embd)))
        



