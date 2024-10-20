from dataclasses import dataclass, asdict
import torch
from typing import Dict, Optional


@dataclass
class RoPEConfig:
    factor: float = 32.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_context_length: int = 8192


@dataclass
class LlamaConfig:
    vocab_size: int = 128_256
    emb_dim: int = 768
    hidden_dim: int = 768
    context_length: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    qkv_bias: bool = False
    n_kv_groups: int = 8
    rope_base: int = 50_000
    rope_freq: RoPEConfig = RoPEConfig()  # Use the RoPEConfig dataclass here
    dtype: Optional[torch.dtype] = torch.bfloat16  # Optional torch dtype with default
