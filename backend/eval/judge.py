"""
backend/eval/judge.py

Faithfulness / groundedness judge.

Given an answer and the contexts it was supposed to use, the judge splits the
answer into atomic claims (sentences) and checks whether each claim is supported
by the contexts. Three backends, in increasing fidelity / cost:

* ``lexical``   — token-containment overlap. Zero dependencies, fully
                  deterministic; the default so the CI gate is reproducible.
* ``embedding`` — cosine similarity of claim vs context sentences (needs an
                  ``embed_fn``; e.g. Ollama nomic-embed-text).
* ``llm``       — an LLM judge (needs a ``chat_fn``; e.g. Ollama llama3) that
                  returns a JSON verdict per answer.

It also scores *completeness* against a fixed report rubric, reused by the
reflection-loop ablation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "is",
    "are", "be", "as", "by", "that", "this", "it", "its", "from", "at", "into",
    "their", "which", "than", "then", "over", "out", "up", "we", "they", "them",
}


def clean_markdown(text: str) -> str:
    text = re.sub(r"\[\d+\]", " ", text)            # citation markers
    text = re.sub(r"[#*`>_]+", " ", text)            # md emphasis / headers
    text = re.sub(r"^\s*[-\d.]+\s+", " ", text, flags=re.M)  # list bullets
    return text


def split_claims(text: str, min_words: int = 4) -> List[str]:
    """Split prose into claim-sized sentences, dropping headings/short fragments."""
    out: List[str] = []
    for raw in _SENT.split(clean_markdown(text)):
        s = raw.strip()
        if len(_WORD.findall(s)) >= min_words:
            out.append(s)
    return out


def _content_tokens(s: str) -> set:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2}


def lexical_overlap(claim: str, context: str) -> float:
    """Containment: fraction of the claim's content tokens present in context."""
    c = _content_tokens(claim)
    if not c:
        return 0.0
    ctx = _content_tokens(context)
    return len(c & ctx) / len(c)


def _cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


@dataclass
class JudgeResult:
    faithfulness: float          # supported claims / total claims
    n_claims: int
    n_supported: int
    unsupported: List[str] = field(default_factory=list)
    per_claim: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "n_claims": self.n_claims,
            "n_supported": self.n_supported,
            "unsupported": self.unsupported,
        }


class GroundednessJudge:
    def __init__(
        self,
        mode: str = "lexical",
        threshold: float = 0.4,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        chat_fn: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        self.mode = mode
        self.threshold = threshold
        self.embed_fn = embed_fn
        self.chat_fn = chat_fn

    # ── public ───────────────────────────────────────────────────────────────
    def score(self, answer: str, contexts: List[str]) -> JudgeResult:
        claims = split_claims(answer)
        if not claims:
            return JudgeResult(0.0, 0, 0)
        if self.mode == "llm" and self.chat_fn is not None:
            return self._score_llm(answer, claims, contexts)
        if self.mode == "embedding" and self.embed_fn is not None:
            return self._score_embedding(claims, contexts)
        return self._score_lexical(claims, contexts)

    # ── backends ───────────────────────────────────────────────────────────────
    def _score_lexical(self, claims: List[str], contexts: List[str]) -> JudgeResult:
        ctx_sents = [s for c in contexts for s in split_claims(c, min_words=3)] or contexts
        per, unsupported = [], []
        for cl in claims:
            best = max((lexical_overlap(cl, cs) for cs in ctx_sents), default=0.0)
            per.append(best)
            if best < self.threshold:
                unsupported.append(cl)
        supported = sum(1 for p in per if p >= self.threshold)
        return JudgeResult(supported / len(claims), len(claims), supported, unsupported, per)

    def _score_embedding(self, claims: List[str], contexts: List[str]) -> JudgeResult:
        ctx_sents = [s for c in contexts for s in split_claims(c, min_words=3)] or contexts
        cvecs = self.embed_fn(ctx_sents)
        qvecs = self.embed_fn(claims)
        per, unsupported = [], []
        for cl, qv in zip(claims, qvecs):
            best = max((_cosine(qv, cv) for cv in cvecs), default=0.0)
            per.append(best)
            if best < self.threshold:
                unsupported.append(cl)
        supported = sum(1 for p in per if p >= self.threshold)
        return JudgeResult(supported / len(claims), len(claims), supported, unsupported, per)

    def _score_llm(self, answer: str, claims: List[str], contexts: List[str]) -> JudgeResult:
        system = (
            "You are a strict faithfulness judge. Given CONTEXT and an ANSWER, decide how many "
            "of the answer's claims are directly supported by the context. "
            'Return ONLY JSON: {"n_claims": <int>, "n_supported": <int>, "unsupported": [<str>]}.'
        )
        user = "CONTEXT:\n" + "\n---\n".join(contexts) + "\n\nANSWER:\n" + answer
        try:
            data = json.loads(self.chat_fn(system, user))
            n = int(data.get("n_claims") or len(claims))
            sup = int(data.get("n_supported") or 0)
            unsup = list(data.get("unsupported") or [])
            n = max(n, 1)
            return JudgeResult(min(sup, n) / n, n, min(sup, n), unsup)
        except Exception:
            return self._score_lexical(claims, contexts)


# ── Completeness rubric (used by the reflection ablation) ───────────────────────
def completeness(report: str) -> float:
    """Fraction of rubric elements present: intro, evidence, citations, conclusion, limitations."""
    low = report.lower()
    checks = [
        len(report.split()) >= 80,                                  # substance
        bool(re.search(r"\[\d+\]", report)),                        # citations
        ("introduction" in low) or ("## " in report),               # structure
        ("conclusion" in low) or ("summary" in low),                # conclusion
        ("limitation" in low) or ("caveat" in low) or ("bias" in low),  # limitations
    ]
    return sum(1 for c in checks if c) / len(checks)
