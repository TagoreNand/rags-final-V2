"""
backend/retrieval/rag.py

Upgraded local RAG retrieval pipeline:

    embed query
      -> ANN candidate generation (hnswlib HNSW, else vectorized brute force)
      -> hybrid fusion of semantic + BM25 rankings via Reciprocal Rank Fusion
      -> second-stage reranking (cross-encoder / LLM / lexical, pluggable)
      -> MMR diversity over the reranked top set (relevance = fusion/rerank score,
         diversity = embedding cosine) so reranking actually drives the result
      -> top-k (text, metadata) blocks

``retrieve_fused`` runs the same ranking for several query variants (multi-query
/ HyDE) and RRF-fuses them before rerank + MMR. The public helpers
``chunk_text``, ``cosine_sim`` and ``mmr`` keep their original signatures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.db import DbPaths, insert_docs, iter_docs
from backend.llm.ollama import OllamaConfig, ollama_embeddings
from backend.retrieval.ann import build_index
from backend.retrieval.query_transform import reciprocal_rank_fusion
from backend.retrieval.rerank import get_reranker
from backend.tracing import get_logger, timed

log = get_logger(__name__)

try:
    from rank_bm25 import BM25Okapi as _BM25

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ── Text utilities ────────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Sentence-aware chunking: tries to break at sentence boundaries within
    [chunk_size-overlap, chunk_size] window. Falls back to hard split when
    text has no sentence boundaries (e.g. repeated characters).
    """
    text = _normalize(text)
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")

    if len(text) <= chunk_size:
        return [text]

    sentences = _SENT_SPLIT.split(text)

    if len(sentences) == 1:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
        return [c for c in chunks if c.strip()]

    chunks = []
    buf = ""
    for sent in sentences:
        candidate = (buf + " " + sent).strip()
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = (buf[-overlap:] + " " + sent).strip() if buf else sent
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


# ── Similarity ────────────────────────────────────────────────────────────────


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def mmr(
    query_vec: np.ndarray,
    candidates: List[Tuple[str, Dict[str, Any], np.ndarray]],
    top_k: int,
    lambda_mult: float = 0.5,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Maximal Marginal Relevance selection with cosine relevance (public helper,
    used directly in tests). Returns diverse top_k results.
    """
    if not candidates:
        return []
    scored = [(t, m, v, cosine_sim(query_vec, v)) for t, m, v in candidates]
    return _mmr_core(scored, top_k, lambda_mult)


def _mmr_core(
    items: List[Tuple[str, Dict[str, Any], np.ndarray, float]],
    top_k: int,
    lambda_mult: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """MMR over items carrying a precomputed relevance score in [0,1]."""
    if not items:
        return []
    selected: List[int] = []
    remaining = list(range(len(items)))
    while len(selected) < min(top_k, len(items)):
        best_idx, best_score = -1, -1e9
        for i in remaining:
            rel = items[i][3]
            if not selected:
                score = rel
            else:
                max_sim = max(cosine_sim(items[s][2], items[i][2]) for s in selected)
                score = lambda_mult * rel - (1 - lambda_mult) * max_sim
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [(items[i][0], items[i][1]) for i in selected]


# ── Config / main class ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class RagConfig:
    top_k: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200
    mmr_lambda: float = 0.6          # 1.0 = pure relevance; 0.0 = pure diversity
    hybrid_alpha: float = 0.7        # retained for reference
    rerank_backend: str = "none"     # none | lexical | llm | cross-encoder
    rrf_k: int = 60                  # Reciprocal Rank Fusion constant
    ann_enabled: bool = True         # use hnswlib ANN when corpus is large enough
    hybrid_enabled: bool = True      # fuse BM25 lexical ranking with semantic via RRF


class LocalRag:
    def __init__(self, db_paths: DbPaths, ollama_cfg: OllamaConfig, rag_cfg: RagConfig):
        self.db_paths = db_paths
        self.ollama_cfg = ollama_cfg
        self.rag_cfg = rag_cfg

    @timed
    def ingest_text(
        self, task_id: Optional[str], text: str, metadata: Dict[str, Any]
    ) -> int:
        chunks = chunk_text(text, self.rag_cfg.chunk_size, self.rag_cfg.chunk_overlap)
        if not chunks:
            return 0
        all_embeddings: List[List[float]] = []
        for i in range(0, len(chunks), 16):
            batch = chunks[i : i + 16]
            all_embeddings.extend(ollama_embeddings(self.ollama_cfg, batch))
        metadatas = [{**metadata, "chunk_index": idx} for idx, _ in enumerate(chunks)]
        insert_docs(
            self.db_paths.db_path,
            task_id=task_id,
            texts=chunks,
            metadatas=metadatas,
            embeddings=all_embeddings,
        )
        log.debug("rag_ingested", n_chunks=len(chunks), source=metadata.get("source"))
        return len(chunks)

    # ── candidate loading ────────────────────────────────────────────────────
    def _load_candidates(
        self, task_id: Optional[str], filter_source: Optional[str]
    ) -> List[Tuple[str, Dict[str, Any], np.ndarray]]:
        cands: List[Tuple[str, Dict[str, Any], np.ndarray]] = []
        for _doc_id, meta, text, vec in iter_docs(self.db_paths.db_path, task_id=task_id):
            if filter_source and meta.get("source") != filter_source:
                continue
            cands.append((text, meta, vec))
        return cands

    # ── single-query candidate ranking: ANN (+ BM25) fused via RRF ────────────
    def _rank_candidates(
        self,
        q_vec: np.ndarray,
        q_text: str,
        candidates: List[Tuple[str, Dict[str, Any], np.ndarray]],
    ) -> List[Tuple[int, float]]:
        n = len(candidates)
        if n == 0:
            return []
        vecs = [c[2] for c in candidates]
        index = build_index(
            vecs, list(range(n)), dim=len(q_vec), prefer_ann=self.rag_cfg.ann_enabled
        )
        depth = min(n, max(self.rag_cfg.top_k * 5, 20))
        semantic = [i for i, _ in index.query(q_vec, depth)]
        rankings = [semantic]

        if self.rag_cfg.hybrid_enabled and _BM25_AVAILABLE and n > 1:
            tokenized = [t.lower().split() for t, _, _ in candidates]
            bm25 = _BM25(tokenized)
            scores = bm25.get_scores(q_text.lower().split())
            bm25_rank = [i for i, _ in sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )]
            rankings.append(bm25_rank)

        return reciprocal_rank_fusion(rankings, k=self.rag_cfg.rrf_k)

    # ── rerank + MMR finishing stage ──────────────────────────────────────────
    def _finish(
        self,
        fused: List[Tuple[int, float]],
        candidates: List[Tuple[str, Dict[str, Any], np.ndarray]],
        query: str,
        rerank: Optional[str],
        chat_fn: Optional[Callable[[str, str], str]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if not fused:
            return []
        backend = rerank if rerank is not None else self.rag_cfg.rerank_backend
        pool = fused[: max(self.rag_cfg.top_k * 4, 12)]
        pool_idx = [i for i, _ in pool]

        # relevance score per candidate index: reranker score, else RRF score
        if backend and backend != "none":
            reranker = get_reranker(backend, chat_fn=chat_fn)
            docs = [(candidates[i][0], candidates[i][1]) for i in pool_idx]
            reranked = reranker.rerank(query, docs, top_n=len(docs))
            idx_by_meta = {id(candidates[i][1]): i for i in pool_idx}
            rel = {idx_by_meta[id(m)]: s for _t, m, s in reranked if id(m) in idx_by_meta}
            scored = [(i, rel.get(i, 0.0)) for i in pool_idx]
        else:
            scored = list(pool)

        # normalise relevance to [0,1] so the MMR relevance/diversity trade-off is balanced
        vals = [s for _, s in scored]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        rng = (hi - lo) or 1.0
        items = [
            (candidates[i][0], candidates[i][1], candidates[i][2], (s - lo) / rng)
            for i, s in scored
        ]
        items.sort(key=lambda x: x[3], reverse=True)
        items = items[: max(self.rag_cfg.top_k * 2, self.rag_cfg.top_k)]
        return _mmr_core(items, self.rag_cfg.top_k, self.rag_cfg.mmr_lambda)

    # ── public retrieval ───────────────────────────────────────────────────────
    @timed
    def retrieve(
        self,
        query: str,
        task_id: Optional[str] = None,
        filter_source: Optional[str] = None,
        rerank: Optional[str] = None,
        chat_fn: Optional[Callable[[str, str], str]] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        q = _normalize(query)
        if not q:
            return []
        candidates = self._load_candidates(task_id, filter_source)
        if not candidates:
            return []
        q_vec = np.array(ollama_embeddings(self.ollama_cfg, [q])[0], dtype=np.float32)
        fused = self._rank_candidates(q_vec, q, candidates)
        return self._finish(fused, candidates, query, rerank, chat_fn)

    @timed
    def retrieve_fused(
        self,
        queries: List[str],
        task_id: Optional[str] = None,
        filter_source: Optional[str] = None,
        rerank: Optional[str] = None,
        chat_fn: Optional[Callable[[str, str], str]] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Multi-query / HyDE retrieval: rank per query, RRF-fuse, rerank + MMR."""
        qs = [_normalize(x) for x in queries if _normalize(x)]
        if not qs:
            return []
        candidates = self._load_candidates(task_id, filter_source)
        if not candidates:
            return []
        embs = ollama_embeddings(self.ollama_cfg, qs)
        per_query_rankings: List[List[int]] = []
        for qt, emb in zip(qs, embs):
            qv = np.array(emb, dtype=np.float32)
            ranked = self._rank_candidates(qv, qt, candidates)
            per_query_rankings.append([i for i, _ in ranked])
        fused = reciprocal_rank_fusion(per_query_rankings, k=self.rag_cfg.rrf_k)
        return self._finish(fused, candidates, qs[0], rerank, chat_fn)
