"""
backend/eval/metrics.py

Standard information-retrieval metrics, implemented as pure functions over
ranked lists of document ids. No dependencies beyond the stdlib, so they are
trivially unit-testable and deterministic.

Conventions
-----------
* ``retrieved`` is the ranked list of doc ids returned by the system (best first).
* ``relevant``  is the set/list of ground-truth relevant doc ids.
* ``k``         truncates the ranked list to its first ``k`` entries.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in rel)
    return hits / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top-k that are relevant."""
    top = retrieved[:k]
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for d in top if d in rel) / len(top)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if at least one relevant doc is in the top-k, else 0.0."""
    rel = set(relevant)
    return 1.0 if any(d in rel for d in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant doc (0 if none retrieved)."""
    rel = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: Sequence[Sequence[str]], relevant_lists: Sequence[Sequence[str]]
) -> float:
    if not retrieved_lists:
        return 0.0
    return sum(
        reciprocal_rank(r, g) for r, g in zip(retrieved_lists, relevant_lists)
    ) / len(retrieved_lists)


def dcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    return sum(
        (1.0 / math.log2(i + 1))
        for i, d in enumerate(retrieved[:k], start=1)
        if d in rel
    )


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalised DCG with binary relevance."""
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = dcg_at_k(retrieved, relevant, k)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def average_precision(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Average precision over the points where a relevant doc is retrieved."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = 0
    score = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            hits += 1
            score += hits / i
    return score / len(rel)


def mean_average_precision(
    retrieved_lists: Sequence[Sequence[str]], relevant_lists: Sequence[Sequence[str]]
) -> float:
    if not retrieved_lists:
        return 0.0
    return sum(
        average_precision(r, g) for r, g in zip(retrieved_lists, relevant_lists)
    ) / len(retrieved_lists)


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of every metric key across per-query rows."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
