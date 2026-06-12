"""
backend/retrieval/grounding.py

Citation-grounding verification.

An LLM is *instructed* to cite sources with [N] markers, but nothing guarantees
the cited source actually supports the sentence. This module closes that loop:
for every sentence that carries an [N] marker, it checks whether the sentence is
supported by the text of the source(s) it cites, and flags:

  * unsupported  — a cited claim whose cited source does NOT support it
                   (mis-citation / hallucinated attribution)
  * uncited      — a substantive factual sentence with NO citation at all

Support is measured by token containment (default, zero-dependency, deterministic)
or embedding cosine (``embed_fn`` provided). Best-effort: never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_CITE = re.compile(r"\[(\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "is",
         "are", "be", "as", "by", "that", "this", "it", "its", "from", "at", "into"}


def _toks(s: str) -> set:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2}


def _containment(claim: str, source: str) -> float:
    c = _toks(claim)
    return len(c & _toks(source)) / len(c) if c else 0.0


def _cosine(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d else 0.0


def _sentences_keep_markers(report: str) -> List[str]:
    body = re.sub(r"^\s*#.*$", "", report, flags=re.M)        # drop headings
    body = re.sub(r"^\s*##\s*References.*", "", body, flags=re.S | re.M)
    body = body.replace("\n", " ")
    return [s.strip() for s in _SENT.split(body) if s.strip()]


@dataclass
class GroundingReport:
    groundedness: float                 # supported cited claims / cited claims
    n_cited_claims: int
    n_supported: int
    unsupported: List[Dict[str, Any]] = field(default_factory=list)
    uncited_claims: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groundedness": round(self.groundedness, 4),
            "n_cited_claims": self.n_cited_claims,
            "n_supported": self.n_supported,
            "n_unsupported": len(self.unsupported),
            "n_uncited": len(self.uncited_claims),
            "unsupported": self.unsupported[:5],
        }


def verify_grounding(
    report: str,
    citations: List[Dict[str, Any]],
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    threshold: float = 0.4,
    min_words: int = 5,
) -> GroundingReport:
    src_text = {
        int(c["num"]): (c.get("snippet") or c.get("text") or "")
        for c in citations
        if "num" in c
    }
    cited, uncited = [], []
    for sent in _sentences_keep_markers(report):
        nums = [int(n) for n in _CITE.findall(sent)]
        clean = _CITE.sub("", sent).strip()
        if len(_WORD.findall(clean)) < min_words:
            continue
        if nums:
            cited.append((clean, nums))
        else:
            uncited.append(clean)

    if not cited:
        return GroundingReport(1.0, 0, 0, [], uncited)

    # score each cited claim against the union of its cited sources
    if embed_fn is not None:
        supported = 0
        unsupported: List[Dict[str, Any]] = []
        for clean, nums in cited:
            srcs = [src_text[n] for n in nums if n in src_text] or [""]
            cvec = embed_fn([clean])[0]
            svecs = embed_fn(srcs)
            best = max((_cosine(cvec, sv) for sv in svecs), default=0.0)
            if best >= threshold:
                supported += 1
            else:
                unsupported.append({"claim": clean, "cited": nums, "score": round(best, 3)})
    else:
        supported = 0
        unsupported = []
        for clean, nums in cited:
            srcs = " ".join(src_text.get(n, "") for n in nums)
            best = _containment(clean, srcs)
            if best >= threshold:
                supported += 1
            else:
                unsupported.append({"claim": clean, "cited": nums, "score": round(best, 3)})

    return GroundingReport(supported / len(cited), len(cited), supported, unsupported, uncited)
