"""
backend/retrieval/query_transform.py

Query-side transformations that improve recall before retrieval:

* ``reciprocal_rank_fusion`` — combine several ranked lists into one robust
  ranking. RRF needs no score calibration (unlike a linear cosine+BM25 blend on
  un-normalized scores), which is why it is used both to fuse hybrid signals and
  to fuse multi-query results.
* ``multi_query`` — ask the LLM for paraphrases / sub-questions so retrieval is
  not hostage to one phrasing. Falls back to ``[goal]`` with no LLM.
* ``hyde`` — Hypothetical Document Embeddings: generate a hypothetical answer and
  embed THAT as the query, which often sits closer to real answer passages.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Any]], k: int = 60
) -> List[Tuple[Any, float]]:
    """Fuse ranked id lists. score(d) = sum 1/(k + rank_in_list(d))."""
    scores: Dict[Any, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def multi_query(
    goal: str, chat_fn: Optional[Callable[[str, str], str]] = None, n: int = 3
) -> List[str]:
    """Return [goal] plus up to n LLM-generated paraphrases (deduped)."""
    queries = [goal.strip()]
    if chat_fn is None:
        return queries
    system = (
        "Rewrite the user's research question into diverse search queries that surface "
        f'different relevant facets. Return ONLY JSON: {{"queries": [<{n} short strings>]}}.'
    )
    try:
        data = json.loads(chat_fn(system, f"Question: {goal}"))
        for q in data.get("queries", []):
            q = str(q).strip()
            if q and q.lower() not in {x.lower() for x in queries}:
                queries.append(q)
    except Exception:
        pass
    return queries[: n + 1]


def hyde(goal: str, chat_fn: Optional[Callable[[str, str], str]] = None) -> str:
    """Generate a short hypothetical answer passage to embed as the query."""
    if chat_fn is None:
        return ""
    system = ("Write a short factual paragraph (3-4 sentences) that would directly "
              "answer the question. Do not hedge. Output only the paragraph.")
    try:
        return re.sub(r"\s+", " ", chat_fn(system, goal)).strip()
    except Exception:
        return ""
