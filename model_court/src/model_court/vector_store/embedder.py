from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HashingEmbedder:
    """
    Deterministic, dependency-free embedder for local runs.
    Produces a sparse-ish hashed bag-of-tokens vector and L2 normalizes it.
    """

    dim: int = 1024

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "little")
            v[h % self.dim] += 1.0
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v /= norm
        return v


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]

