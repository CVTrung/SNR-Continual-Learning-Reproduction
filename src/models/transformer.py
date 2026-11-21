"""Single-layer decoder Transformer for Permuted Shakespeare.

Status: executable.
Architecture (paper Section 3.1): one decoder block, 2 heads of dimension 16 (model width 32),
a feed-forward layer of 256 ReLU units, GPT-2 BPE vocabulary restricted to the tokens in the
sampled text, context 128.
Team decisions (the paper does not state them; docs/reproduction_notes.md):
- GPT-style pre-LayerNorm block with residual connections, learned positional embeddings,
  a final LayerNorm and an untied output head.
- Separate Q, K, V projections so that the Q and K weight norms of Figure 4 can be measured.
- Linear layers use the same LeCun-normal init as the MLP; embeddings use N(0, 0.02^2).
Interventions: `reset_groups()` exposes only the FFN (fc1 -> fc2), because attention heads are
not neurons. `attention_parameters()` is the mask of the L2* (attention-only L2) ablation.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mlp import init_linear_


class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_ffn, context):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, n_ffn)
        self.fc2 = nn.Linear(n_ffn, d_model)
        mask = torch.tril(torch.ones(context, context, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        b, t, d = x.shape
        h = self.ln1(x)

        def heads(z):
            return z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)

        q, k, v = heads(self.q(h)), heads(self.k(h)), heads(self.v(h))
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.n_heads)
        scores = scores.masked_fill(~self.mask[:t, :t], float("-inf"))
        probs = scores.softmax(dim=-1)                      # (B, heads, T, T)
        att = (probs @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(att)
        ffn = torch.relu(self.fc1(self.ln2(x)))             # (B, T, n_ffn)
        x = x + self.fc2(ffn)
        return x, ffn, probs


class DecoderTransformer(nn.Module):
    """Decoder-only language model; forward(x) returns logits (B, T, vocab).

    Parameters: vocab_size, context, d_model (32), n_heads (2), n_ffn (256).
    """

    def __init__(self, vocab_size, context=128, d_model=32, n_heads=2, n_ffn=256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.block = Block(d_model, n_heads, n_ffn, context)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        for layer in (self.block.q, self.block.k, self.block.v, self.block.proj,
                      self.block.fc1, self.block.fc2, self.head):
            init_linear_(layer)

    def forward(self, x, return_activations=False, return_attention=False):
        t = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(t, device=x.device))
        h, ffn, probs = self.block(h)
        logits = self.head(self.ln_f(h))
        if not (return_activations or return_attention):
            return logits
        out = [logits]
        if return_activations:
            out.append([ffn])
        if return_attention:
            out.append(probs)
        return tuple(out)

    def reset_groups(self):
        """Only the feed-forward units are resettable neurons."""
        return [(self.block.fc1, self.block.fc2)]

    def attention_parameters(self):
        """Q, K, V and output-projection weights and biases (mask for L2*)."""
        b = self.block
        return [p for layer in (b.q, b.k, b.v, b.proj) for p in layer.parameters()]

    def ffn_parameters(self):
        b = self.block
        return [*b.fc1.parameters(), *b.fc2.parameters()]
