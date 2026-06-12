"""
backend/eval/ablation.py

Reflection-loop A/B ablation: does the self-critique loop earn its extra LLM
cost? For every golden item we score two answers with the SAME judge:

  * draft       — a single synthesis pass (no reflection)
  * reflected   — after the Reflector enforces the rubric (citations, evidence,
                  limitations, structure) and the Analyst revises

and report faithfulness, rubric completeness, and the LLM-call cost of each.

Offline (default) it is a *controlled* ablation: the draft is a terse first
pass and the reflected answer is the rubric-complete version, so the harness
demonstrates the measurement and the expected direction reproducibly. Pass real
``draft_fn``/``revise_fn`` (e.g. the pipeline's _synthesize/_reflect against
Ollama) for production magnitudes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from backend.eval import golden as G
from backend.eval.judge import GroundednessJudge, completeness


@dataclass
class AblationRow:
    label: str
    faithfulness: float
    completeness: float
    llm_calls: int


_LIMIT_HINTS = ("bottleneck", "hacking", "limitation", "prevent", "penalty",
                "bias", "ceiling", "cheaper")


def _draft(item: G.GoldenItem) -> str:
    """Terse first pass: grounded but no citations, structure, or limitations."""
    return item.reference_answer


def _reflected(item: G.GoldenItem) -> str:
    """
    Rubric-complete answer the reflection loop drives toward. Evidence and the
    limitation are extracted VERBATIM from the cited sources, so the longer
    answer stays grounded (faithfulness should not regress) while completeness
    rises — exactly the trade the reflector is meant to make.
    """
    docs = [G.CORPUS_BY_ID[u] for u in item.relevant if u in G.CORPUS_BY_ID]
    intro = f"{item.reference_answer} [1]"
    evidence = ""
    if docs:
        evidence = docs[0].text.split(". ")[0].strip().rstrip(".") + " [1]."
    limitation = ""
    for d in docs:
        for sent in d.text.split(". "):
            if any(h in sent.lower() for h in _LIMIT_HINTS):
                limitation = sent.strip().rstrip(".") + " [1]."
                break
        if limitation:
            break
    if not limitation:
        limitation = evidence or f"{item.reference_answer} [1]"
    return (
        f"## Introduction\n{intro}\n\n"
        f"## Evidence\n{evidence}\n\n"
        f"## Limitations\n{limitation}\n\n"
        f"## Conclusion\nIn summary, {intro}"
    )


def reflection_ablation(
    judge: Optional[GroundednessJudge] = None,
    items: Optional[List[G.GoldenItem]] = None,
    draft_fn: Optional[Callable[[G.GoldenItem], str]] = None,
    revise_fn: Optional[Callable[[G.GoldenItem], str]] = None,
    reflections: int = 2,
) -> List[AblationRow]:
    judge = judge or GroundednessJudge(mode="lexical")
    items = items or G.GOLDEN
    draft_fn = draft_fn or _draft
    revise_fn = revise_fn or _reflected

    df, dc, rf, rc = [], [], [], []
    for it in items:
        ctx = [G.CORPUS_BY_ID[u].text for u in it.relevant if u in G.CORPUS_BY_ID]
        d, r = draft_fn(it), revise_fn(it)
        df.append(judge.score(d, ctx).faithfulness)
        dc.append(completeness(d))
        rf.append(judge.score(r, ctx).faithfulness)
        rc.append(completeness(r))

    n = len(items)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return [
        AblationRow("no reflection (draft)", mean(df), mean(dc), n * 1),
        AblationRow(f"reflection (<=x{reflections})", mean(rf), mean(rc),
                    n * (1 + 2 * reflections)),
    ]


def format_table(rows: List[AblationRow]) -> str:
    out = ["", "  Reflection-loop A/B ablation",
           "  " + "-" * 60,
           f"  {'config':<26}{'faithful':>10}{'complete':>10}{'llm_calls':>11}",
           "  " + "-" * 60]
    for r in rows:
        out.append(f"  {r.label:<26}{r.faithfulness:>10.3f}{r.completeness:>10.3f}{r.llm_calls:>11}")
    if len(rows) == 2:
        d, r = rows
        out.append("  " + "-" * 60)
        out.append(f"  {'delta':<26}{r.faithfulness-d.faithfulness:>+10.3f}"
                   f"{r.completeness-d.completeness:>+10.3f}{r.llm_calls-d.llm_calls:>+11}")
        verdict = "earns its cost" if (r.completeness - d.completeness) > 0.05 else "marginal"
        out.append("  " + "-" * 60)
        out.append(f"  verdict: reflection {verdict} "
                   f"(+{r.completeness-d.completeness:.0%} completeness for "
                   f"{r.llm_calls-d.llm_calls} extra LLM calls)")
    return "\n".join(out)
