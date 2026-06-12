"""
backend/eval — offline evaluation harness.

Exposes retrieval metrics, the golden Q/A dataset, and the groundedness judge.
The harness (backend.eval.harness) wires these into a reproducible quality gate
that runs in CI without requiring Ollama or a GPU.
"""
from backend.eval.metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    average_precision,
    mean_average_precision,
    aggregate_metrics,
)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "average_precision",
    "mean_average_precision",
    "aggregate_metrics",
]
