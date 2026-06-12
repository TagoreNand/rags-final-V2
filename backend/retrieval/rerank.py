"""
backend/retrieval/rerank.py

Pluggable second-stage reranker. First-stage retrieval optimizes for recall
(grab a broad candidate set cheaply); the reranker optimizes for precision
(reorder those candidates by true query-document relevance).

Backends (graceful fallback in this order if a dependency is missing):

* ``cross-encoder`` — a sentence-transformers CrossEncoder that jointly encodes
                      (query, doc) pairs. The highest-fidelity option; optional
                      dependency.
* ``llm``           — pointwise LLM relevance scoring via Ollama (``chat_fn``).
* ``lexical``       — deterministic query/doc token-overlap. Zero-dependency
                      default, also used for the reproducible eval gate.
* ``none``          — passthrough (keeps first-stage order).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.tracing import get_logger

log = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with",
         "is", "are", "be", "as", "by", "that", "this", "it", "its", "from"}

Doc = Tuple[str, Dict[str, Any]]
Scored = Tuple[str, Dict[str, Any], float]


def _toks(s: str) -> set:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2}


class NoopReranker:
    backend = "none"

    def rerank(self, query: str, docs: List[Doc], top_n: int) -> List[Scored]:
        return [(t, m, 0.0) for t, m in docs][:top_n]


class LexicalReranker:
    """Deterministic: score = containment of query tokens in the document."""

    backend = "lexical"

    def rerank(self, query: str, docs: List[Doc], top_n: int) -> List[Scored]:
        q = _toks(query)
        scored = []
        for t, m in docs:
            d = _toks(t)
            overlap = len(q & d) / len(q) if q else 0.0
            title_bonus = 0.15 if q & _toks(m.get("title", "")) else 0.0
            scored.append((t, m, min(1.0, overlap + title_bonus)))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]


class LLMReranker:
    """Pointwise LLM relevance scoring (0-1) per candidate via chat_fn."""

    backend = "llm"

    def __init__(self, chat_fn: Callable[[str, str], str]) -> None:
        self.chat_fn = chat_fn

    def rerank(self, query: str, docs: List[Doc], top_n: int) -> List[Scored]:
        sys = ("Rate how relevant the document is to the query on a 0-100 scale. "
               "Return ONLY the integer.")
        scored = []
        for t, m in docs:
            try:
                raw = self.chat_fn(sys, f"Query: {query}\n\nDocument: {t[:600]}")
                val = float(re.findall(r"\d+", raw)[0]) / 100.0
            except Exception:
                val = 0.0
            scored.append((t, m, max(0.0, min(1.0, val))))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder (optional dependency)."""

    backend = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # optional import

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Doc], top_n: int) -> List[Scored]:
        if not docs:
            return []
        scores = self._model.predict([(query, t[:600]) for t, _ in docs])
        ranked = sorted(
            ((t, m, float(s)) for (t, m), s in zip(docs, scores)),
            key=lambda x: x[2], reverse=True,
        )
        return ranked[:top_n]


def get_reranker(backend: str = "lexical",
                 chat_fn: Optional[Callable[[str, str], str]] = None):
    """Factory with graceful degradation to a working backend."""
    backend = (backend or "none").lower()
    if backend == "none":
        return NoopReranker()
    if backend == "cross-encoder":
        try:
            return CrossEncoderReranker()
        except Exception as exc:  # pragma: no cover - depends on optional dep
            log.warning("rerank_fallback", wanted="cross-encoder", error=str(exc))
            backend = "lexical"
    if backend == "llm":
        if chat_fn is not None:
            return LLMReranker(chat_fn)
        log.warning("rerank_fallback", wanted="llm", error="no chat_fn")
        backend = "lexical"
    return LexicalReranker()
