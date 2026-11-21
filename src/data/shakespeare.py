"""Load The Tempest as GPT-2 BPE tokens over a restricted vocabulary (Permuted Shakespeare).

Status: executable.
Inputs: the complete-works text file prepared by scripts/prepare_data.py. It matches the
authors' reference file src/data/all_shakespeare.txt byte for byte after CRLF->LF
normalization (checked 2026-11-21, sha256 4e81a0f2...).
Outputs: sequences (N, context + 1) of token ids in [0, V) and the GPT-2 id of each of the V
tokens.
Paper protocol: 32768 tokens of The Tempest, GPT-2 BPE "limited to the set of unique tokens
present in the sampled text", batch 8, context 128.
Team decisions (docs/reproduction_notes.md):
- The play runs from "THE TEMPEST / by William Shakespeare" to its "THE END". Newlines are
  kept, and only CRLF is normalized. The authors' public template instead reads the first
  tokens of the whole file, which is a different play, so we follow the paper's text.
- 256 non-overlapping windows of 129 tokens (33,024 tokens). An epoch is then exactly
  32 batches of 8 x 128 next-token predictions, which is the paper's 32,768 tokens.
- The vocabulary is the sorted set of GPT-2 ids in those 33,024 tokens.
"""
from pathlib import Path

import numpy as np

START = "THE TEMPEST\n\nby William Shakespeare"
END = "THE END"


def extract_tempest(text):
    """Return The Tempest (title to closing 'THE END') from the complete-works text."""
    text = text.replace("\r\n", "\n")
    start = text.find(START)
    if start < 0:
        raise ValueError("The Tempest header not found")
    end = text.find(END, start)
    if end < 0:
        raise ValueError("end of The Tempest not found")
    return text[start:end + len(END)]


def remap_vocabulary(tokens):
    """Map token ids onto 0..V-1 (sorted). Returns (ids, vocab) with vocab[ids] == tokens."""
    vocab, ids = np.unique(np.asarray(tokens), return_inverse=True)
    return ids.astype(np.int64), vocab


def make_sequences(tokens, context=128, n_sequences=256):
    """Cut the first n_sequences * (context + 1) tokens into non-overlapping windows."""
    need = n_sequences * (context + 1)
    tokens = np.asarray(tokens)
    if len(tokens) < need:
        raise ValueError(f"need {need} tokens, got {len(tokens)}")
    return tokens[:need].reshape(n_sequences, context + 1)


def load_tempest(path, context=128, batch_size=8, batches_per_epoch=32):
    """Tokenize The Tempest and return (sequences (N, context+1) int64, vocab GPT-2 ids)."""
    import tiktoken

    text = Path(path).read_text(encoding="utf-8")
    enc = tiktoken.get_encoding("gpt2")
    n_seq = batch_size * batches_per_epoch
    raw = make_sequences(enc.encode(extract_tempest(text)), context, n_seq)
    ids, vocab = remap_vocabulary(raw.reshape(-1))
    return ids.reshape(raw.shape), vocab
