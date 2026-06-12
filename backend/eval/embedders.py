"""
backend/eval/embedders.py

Pluggable embedders for the eval harness.

* ``LexicalEmbedder`` — a deterministic hashing embedder (word tokens + char
  trigrams hashed into a fixed-dim vector, L2-normalised). No model, no network,
  fully reproducible — so the CI gate produces stable retrieval numbers without
  Ollama. It captures lexical/morphological overlap, which is enough to exercise
  and regression-test the retrieval stack.
* ``OllamaEmbedder`` — thin adapter over the production nomic-embed-text path for
  high-fidelity local runs (``--embedder ollama``).

Both expose ``embed(texts) -> List[List[float]]`` and are callable, so they can
be injected wherever the code expects ``ollama_embeddings(cfg, texts)``.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

_WORD = re.compile(r"[a-z0-9]+")


class LexicalEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> List[str]:
        toks = _WORD.findall(text.lower())
        feats = list(toks)
        joined = " ".join(toks)
        for i in range(len(joined) - 2):  # char trigrams
            feats.append("#" + joined[i : i + 3])
        return feats

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for f in self._features(text):
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            v[idx] += sign
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)


class OllamaEmbedder:
    def __init__(self, cfg) -> None:
        from backend.llm.ollama import OllamaConfig

        self._cfg = cfg if isinstance(cfg, OllamaConfig) else None
        self._raw = cfg

    def embed(self, texts: List[str]) -> List[List[float]]:
        from backend.llm.ollama import ollama_embeddings

        return ollama_embeddings(self._cfg or self._raw, texts)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)
