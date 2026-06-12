"""
backend/retrieval/ann.py

Approximate-nearest-neighbour index with a vectorized brute-force fallback.

The previous retrieval path scored candidates one-at-a-time in a Python loop
(`cosine_sim` per doc). This module replaces that with either:

* ``HnswIndex``       — hnswlib HNSW graph (sub-linear ANN) when hnswlib is
                        installed; the right choice as the corpus grows.
* ``BruteForceIndex`` — a single vectorized NumPy matmul (exact, O(n) but ~100x
                        faster than the per-item loop). Always available.

Both expose ``query(vector, k) -> List[(id, score)]`` with cosine similarity in
``[0, 1]`` (higher is better), so callers are backend-agnostic.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:
    import hnswlib  # type: ignore

    _HNSW_AVAILABLE = True
except Exception:  # pragma: no cover
    _HNSW_AVAILABLE = False


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class BruteForceIndex:
    """Exact cosine search via one vectorized matmul."""

    backend = "bruteforce"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._mat = np.empty((0, dim), dtype=np.float32)
        self._ids: List[int] = []

    def build(self, vectors: Sequence[Sequence[float]], ids: Sequence[int]) -> None:
        self._mat = _normalize(np.asarray(vectors, dtype=np.float32))
        self._ids = list(ids)

    def query(self, vector: Sequence[float], k: int) -> List[Tuple[int, float]]:
        if self._mat.shape[0] == 0:
            return []
        q = np.asarray(vector, dtype=np.float32)
        n = np.linalg.norm(q) or 1.0
        sims = self._mat @ (q / n)
        k = min(k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._ids[i], float(sims[i])) for i in top]


class HnswIndex:
    """hnswlib HNSW index (cosine space)."""

    backend = "hnsw"

    def __init__(self, dim: int, ef: int = 64, M: int = 16) -> None:
        self.dim = dim
        self.ef = ef
        self.M = M
        self._index = None
        self._ids: List[int] = []

    def build(self, vectors: Sequence[Sequence[float]], ids: Sequence[int]) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        n = max(len(vecs), 1)
        idx = hnswlib.Index(space="cosine", dim=self.dim)
        idx.init_index(max_elements=n, ef_construction=200, M=self.M)
        idx.add_items(vecs, np.arange(n))
        idx.set_ef(max(self.ef, 16))
        self._index = idx
        self._ids = list(ids)

    def query(self, vector: Sequence[float], k: int) -> List[Tuple[int, float]]:
        if self._index is None or not self._ids:
            return []
        k = min(k, len(self._ids))
        labels, distances = self._index.knn_query(
            np.asarray(vector, dtype=np.float32), k=k
        )
        # hnswlib cosine "distance" = 1 - cosine_similarity
        return [
            (self._ids[int(lbl)], 1.0 - float(dist))
            for lbl, dist in zip(labels[0], distances[0])
        ]


def build_index(
    vectors: Sequence[Sequence[float]],
    ids: Sequence[int],
    dim: int,
    prefer_ann: bool = True,
):
    """Build the best available index for the given vectors."""
    use_hnsw = prefer_ann and _HNSW_AVAILABLE and len(vectors) >= 64
    index = HnswIndex(dim) if use_hnsw else BruteForceIndex(dim)
    index.build(vectors, ids)
    return index
