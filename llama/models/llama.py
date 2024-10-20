import torch
import torch.nn as nn
import torch.nn.functional as F
from llama.models._modules import *
from typing import Dict, Any
from typing import Optional
from llama.models.utils import LlamaConfig


class Llama3Model(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        self.cfg = cfg

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(
            cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype
        )

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds  # + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        """
        Generate new tokens based on the given input indices.

        This function generates new tokens for the input sequence up to a specified
        number of new tokens. The process involves adjusting the logits using the
        provided temperature and optionally applying top-k sampling.

        :param idx: A tensor of shape (batch_size, sequence_length) containing the input indices.
        :type idx: torch.Tensor
        :param max_new_tokens: The maximum number of new tokens to generate.
        :type max_new_tokens: int
        :param temperature: The temperature value for scaling the logits. Default is 1.0.
        :type temperature: float
        :param top_k: The number of top logits to consider for sampling. Default is 50.
        :type top_k: Optional[int]
        :return: A tensor containing the input indices concatenated with the generated new tokens.
        :rtype: Tensor
        """

        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length :]

            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            assert temperature >= 0.0

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
