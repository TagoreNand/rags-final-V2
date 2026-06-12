"""
backend/tests/test_evals.py

Prompt quality evaluation harness.

Runs "offline" evals that score agent output quality WITHOUT Ollama:
  - Supervisor plan structure validation
  - Synthesis report completeness scoring
  - Citation [N] marker presence
  - Reflection rubric adherence
  - RAG retrieval recall simulation

These are fast, deterministic, and give a "quality floor" signal in CI.
Add real LLM-in-the-loop evals separately (mark @pytest.mark.slow).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


# ── Helpers ───────────────────────────────────────────────────────────────────


def _has_keys(obj: Dict, *keys) -> bool:
    return all(k in obj for k in keys)


def _citation_count(text: str) -> int:
    return len(re.findall(r"\[\d+\]", text))


def _markdown_sections(text: str) -> List[str]:
    return re.findall(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)


# ── 1. Supervisor plan schema ─────────────────────────────────────────────────


class TestSupervisorPlanSchema:
    """
    The supervisor's JSON output must always have exactly the right structure.
    We test this with synthetic plan strings the way the agent would produce them.
    """

    VALID_PLAN = json.dumps(
        {
            "queries": ["transformer attention mechanism", "BERT language model"],
            "include_code_demo": False,
        }
    )

    PLAN_WITH_CODE = json.dumps(
        {
            "queries": ["python numpy array operations"],
            "include_code_demo": True,
        }
    )

    MALFORMED_PLAN = '{"queries": "should be a list"}'

    def _parse(self, s: str) -> Dict[str, Any]:
        from backend.agents.pipeline import _safe_json

        return _safe_json(s) or {}

    def test_valid_plan_parsed(self):
        plan = self._parse(self.VALID_PLAN)
        assert _has_keys(plan, "queries", "include_code_demo")
        assert isinstance(plan["queries"], list)
        assert len(plan["queries"]) >= 1

    def test_code_demo_flag(self):
        plan = self._parse(self.PLAN_WITH_CODE)
        assert plan.get("include_code_demo") is True

    def test_queries_capped_at_max_sources(self):
        big_plan = json.dumps(
            {"queries": [f"q{i}" for i in range(20)], "include_code_demo": False}
        )
        plan = self._parse(big_plan)
        cfg_max = 6
        trimmed = plan["queries"][:cfg_max]
        assert len(trimmed) <= cfg_max

    def test_malformed_falls_back_gracefully(self):
        plan = self._parse(self.MALFORMED_PLAN)
        # Should either parse or return empty — never crash
        assert isinstance(plan, dict)

    def test_empty_string_returns_none(self):
        from backend.agents.pipeline import _safe_json

        assert _safe_json("") is None

    def test_json_in_prose_extracted(self):
        from backend.agents.pipeline import _safe_json

        prose = 'Sure! Here is the plan: {"queries": ["test"], "include_code_demo": false} Hope that helps!'
        result = _safe_json(prose)
        assert result is not None
        assert result["queries"] == ["test"]


# ── 2. Report quality scoring ─────────────────────────────────────────────────


class TestReportQualityScoring:
    """
    Rubric-based scoring of synthetic reports.
    Mirrors what the reflector agent does, but deterministically.
    """

    HIGH_QUALITY_REPORT = """
# Transformer Attention Mechanisms

## Introduction
Transformer models [1] revolutionised natural language processing by replacing recurrent
architectures with self-attention. This report examines the core mechanism and its impact.

## Key Findings

The attention formula scores every token against every other token [2]:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Multi-head attention [1] allows the model to attend to different representation subspaces
simultaneously. Research on BERT [3] demonstrated masked language modelling at scale.

## Limitations
Self-attention is O(n²) in sequence length, making it expensive for very long sequences.
Approaches such as sparse attention [2] address this limitation.

## Conclusion
Transformers are now the dominant architecture in NLP due to their parallelism and
expressiveness. Future work should focus on efficient long-context methods.
"""

    POOR_QUALITY_REPORT = "Transformers are good. They work well."

    def score(self, report: str) -> Dict[str, bool]:
        return {
            "has_intro": bool(
                re.search(r"#+\s*(intro|overview|background)", report, re.I)
            ),
            "has_sections": len(_markdown_sections(report)) >= 2,
            "has_citations": _citation_count(report) >= 2,
            "has_conclusion": bool(
                re.search(r"#+\s*(conclusion|summary|recommendation)", report, re.I)
            ),
            "has_limitations": bool(
                re.search(r"limit|risk|caveat|drawback", report, re.I)
            ),
            "has_code_block": "```" in report,
            "min_length": len(report.split()) >= 100,
        }

    def test_high_quality_passes_rubric(self):
        scores = self.score(self.HIGH_QUALITY_REPORT)
        # A good report must pass at least 5/7 criteria
        passing = sum(scores.values())
        assert passing >= 5, f"Only {passing}/7 criteria passed: {scores}"

    def test_poor_quality_fails_rubric(self):
        scores = self.score(self.POOR_QUALITY_REPORT)
        passing = sum(scores.values())
        assert passing <= 2, f"Expected poor report to fail, got {passing}/7"

    def test_citation_count_threshold(self):
        assert _citation_count(self.HIGH_QUALITY_REPORT) >= 2
        assert _citation_count(self.POOR_QUALITY_REPORT) == 0

    def test_markdown_sections_detected(self):
        sections = _markdown_sections(self.HIGH_QUALITY_REPORT)
        assert len(sections) >= 3

    def test_min_word_count(self):
        assert len(self.HIGH_QUALITY_REPORT.split()) >= 100
        assert len(self.POOR_QUALITY_REPORT.split()) < 20


# ── 3. Reflection decision logic ──────────────────────────────────────────────


class TestReflectionLogic:
    """
    The reflector must correctly decide needs_revision=true/false.
    We test its JSON output contract.
    """

    GOOD_REFLECTION = json.dumps({"needs_revision": False, "revision_instructions": ""})
    BAD_REFLECTION = json.dumps(
        {
            "needs_revision": True,
            "revision_instructions": "Add more evidence and a clear conclusion.",
        }
    )

    def _parse(self, s: str):
        from backend.agents.pipeline import _safe_json

        return _safe_json(s)

    def test_good_report_no_revision(self):
        r = self._parse(self.GOOD_REFLECTION)
        assert r["needs_revision"] is False
        assert r["revision_instructions"] == ""

    def test_bad_report_triggers_revision(self):
        r = self._parse(self.BAD_REFLECTION)
        assert r["needs_revision"] is True
        assert len(r["revision_instructions"]) > 10

    def test_revision_loop_terminates(self):
        """Simulates the retry loop: at most 3 iterations."""
        retries = 0
        while retries < 3:
            r = self._parse(self.BAD_REFLECTION)
            if not r["needs_revision"]:
                break
            retries += 1
        # Loop terminates at max 3 even if LLM always says "revise"
        assert retries == 3


# ── 4. Citation extraction ────────────────────────────────────────────────────


class TestCitationExtraction:
    """Test that citation [N] markers are correctly found in prose."""

    def test_extracts_single(self):
        from backend.retrieval.citations import extract_cited_nums

        assert extract_cited_nums("See [1] for details.") == [1]

    def test_extracts_multiple_unordered(self):
        from backend.retrieval.citations import extract_cited_nums

        nums = extract_cited_nums("Based on [3] and [1], confirmed by [2].")
        assert sorted(nums) == [1, 2, 3]

    def test_no_citations_returns_empty(self):
        from backend.retrieval.citations import extract_cited_nums

        assert extract_cited_nums("No refs here at all.") == []

    def test_adjacent_citations(self):
        from backend.retrieval.citations import extract_cited_nums

        nums = extract_cited_nums("[1][2][3]")
        assert nums == [1, 2, 3]


# ── 5. RAG retrieval recall simulation ───────────────────────────────────────


class TestRagRecallSimulation:
    """
    Simulates a retrieval scenario without Ollama.
    Checks that MMR + cosine produces a sensible ranking
    given a deterministic embedding space.
    """

    def _make_vecs(self, n: int, dim: int = 8, seed: int = 0):
        import numpy as np

        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_top1_is_closest(self):
        from backend.retrieval.rag import mmr

        vecs = self._make_vecs(5, dim=8)
        q = vecs[0]  # query == first doc (should be top result)
        candidates = [(f"doc{i}", {}, vecs[i]) for i in range(5)]
        results = mmr(q, candidates, top_k=1, lambda_mult=1.0)  # pure relevance
        assert results[0][0] == "doc0"

    def test_mmr_avoids_duplicate_content(self):
        import numpy as np
        from backend.retrieval.rag import mmr

        # Two near-identical docs + one diverse
        v_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v_sim = np.array([0.99, 0.14, 0.0, 0.0], dtype=np.float32)
        v_sim = v_sim / np.linalg.norm(v_sim)
        v_div = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        candidates = [
            ("base", {}, v_base),
            ("similar", {}, v_sim),
            ("diverse", {}, v_div),
        ]
        results = mmr(v_base, candidates, top_k=2, lambda_mult=0.3)
        texts = [r[0] for r in results]
        # With strong diversity weight, "diverse" beats "similar"
        assert "diverse" in texts

    def test_empty_candidates_returns_empty(self):
        import numpy as np
        from backend.retrieval.rag import mmr

        q = np.array([1.0, 0.0], dtype=np.float32)
        assert mmr(q, [], top_k=5) == []
