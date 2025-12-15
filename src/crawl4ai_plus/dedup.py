"""Deduplication helpers using content hash and lightweight simhash/minhash."""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, List, Set, Tuple


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def simhash(text: str, bits: int = 64) -> int:
    tokens = _tokenize(text)
    if not tokens:
        return 0
    v = [0] * bits
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            bitmask = 1 << i
            v[i] += 1 if h & bitmask else -1
    fingerprint = 0
    for i in range(bits):
        if v[i] >= 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def minhash_signature(tokens: Iterable[str], num_hashes: int = 32) -> Tuple[int, ...]:
    sig = []
    token_list = list(tokens)
    if not token_list:
        return tuple()
    for seed in range(num_hashes):
        random.seed(seed)
        mins = min((hashlib.sha1((t + str(seed)).encode("utf-8")).hexdigest() for t in token_list))
        sig.append(int(mins, 16))
    return tuple(sig)


def jaccard_from_minhash(sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


class Deduper:
    """Deduplication registry using content hash and simhash."""

    def __init__(self, simhash_threshold: int = 5, jaccard_threshold: float = 0.82):
        self.hashes: Set[str] = set()
        self.simhashes: List[int] = []
        self.simhash_threshold = simhash_threshold
        self.minsigs: List[Tuple[int, ...]] = []
        self.jaccard_threshold = jaccard_threshold

    def seen(self, text: str) -> bool:
        if not text:
            return False
        h = content_hash(text)
        if h in self.hashes:
            return True
        sh = simhash(text)
        for existing in self.simhashes:
            if hamming_distance(existing, sh) <= self.simhash_threshold:
                return True
        sig = minhash_signature(_tokenize(text))
        for existing_sig in self.minsigs:
            if jaccard_from_minhash(existing_sig, sig) >= self.jaccard_threshold:
                return True
        self.hashes.add(h)
        self.simhashes.append(sh)
        self.minsigs.append(sig)
        return False


__all__ = ["content_hash", "simhash", "hamming_distance", "minhash_signature", "jaccard_from_minhash", "Deduper"]
