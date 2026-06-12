"""
Tests for the evaluation harness and the advanced retrieval stack:
metrics math, groundedness judge, deterministic embedder, RRF, ANN index,
reranker backends, citation grounding, query transforms, the harness gate,
and the reflection ablation. All deterministic and offline.
"""
import numpy as np
import pytest


# ── Retrieval metrics ───────────────────────────────────────────────────────
class TestMetrics:
    def test_recall_at_k(self):
        from backend.eval.metrics import recall_at_k
        assert recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5
        assert recall_at_k(["a", "b"], ["x"], 2) == 0.0
        assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0

    def test_precision_at_k(self):
        from backend.eval.metrics import precision_at_k
        assert precision_at_k(["a", "b", "c"], ["a", "b"], 2) == 1.0
        assert precision_at_k(["x", "y"], ["a"], 2) == 0.0

    def test_reciprocal_rank(self):
        from backend.eval.metrics import reciprocal_rank
        assert reciprocal_rank(["x", "a", "y"], ["a"]) == pytest.approx(0.5)
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_ndcg_perfect_vs_worst(self):
        from backend.eval.metrics import ndcg_at_k
        rel = ["a", "b"]
        assert ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)
        assert ndcg_at_k(["c", "d", "a"], rel, 3) < ndcg_at_k(["a", "b", "c"], rel, 3)

    def test_average_precision(self):
        from backend.eval.metrics import average_precision
        assert average_precision(["a", "x", "b"], ["a", "b"]) == pytest.approx((1.0 + 2/3) / 2)

    def test_aggregate(self):
        from backend.eval.metrics import aggregate_metrics
        agg = aggregate_metrics([{"r": 1.0}, {"r": 0.0}])
        assert agg["r"] == 0.5


# ── Groundedness judge ──────────────────────────────────────────────────────
class TestJudge:
    def test_grounded_claim_scores_high(self):
        from backend.eval.judge import GroundednessJudge
        j = GroundednessJudge(mode="lexical", threshold=0.4)
        ctx = ["Proximal Policy Optimization uses a KL penalty to constrain the policy."]
        ans = "RLHF uses Proximal Policy Optimization with a KL penalty to constrain the policy."
        assert j.score(ans, ctx).faithfulness == 1.0

    def test_hallucinated_claim_flagged(self):
        from backend.eval.judge import GroundednessJudge
        j = GroundednessJudge(mode="lexical", threshold=0.4)
        res = j.score("The Eiffel Tower was built by penguins in Antarctica.",
                      ["Reinforcement learning optimizes a reward model."])
        assert res.faithfulness == 0.0
        assert res.unsupported

    def test_completeness_rubric(self):
        from backend.eval.judge import completeness
        bare = "RLHF aligns models."
        full = ("## Introduction\nRLHF aligns models with human feedback [1]. " * 10 +
                "\n## Conclusion\nIn summary a key limitation is reward hacking [2].")
        assert completeness(full) > completeness(bare)


# ── Deterministic embedder ──────────────────────────────────────────────────
class TestEmbedder:
    def test_deterministic_and_normalized(self):
        from backend.eval.embedders import LexicalEmbedder
        e = LexicalEmbedder(dim=256)
        v1, v2 = e.embed(["hello world"])[0], e.embed(["hello world"])[0]
        assert v1 == v2
        assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-6)

    def test_related_more_similar_than_unrelated(self):
        from backend.eval.embedders import LexicalEmbedder
        from backend.retrieval.rag import cosine_sim
        e = LexicalEmbedder()
        a, b, c = e.embed(["reward model human preference",
                           "a reward model predicts human preference",
                           "the eiffel tower is in paris"])
        assert cosine_sim(np.array(a), np.array(b)) > cosine_sim(np.array(a), np.array(c))


# ── RRF + ANN + reranker ────────────────────────────────────────────────────
class TestFusionAnnRerank:
    def test_rrf_rewards_consensus(self):
        from backend.retrieval.query_transform import reciprocal_rank_fusion
        fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]]))
        assert max(fused, key=fused.get) == "a"

    def test_bruteforce_index_orders_by_cosine(self):
        from backend.retrieval.ann import BruteForceIndex
        idx = BruteForceIndex(dim=2)
        idx.build([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], [10, 20, 30])
        top = idx.query([1.0, 0.0], 2)
        assert top[0][0] == 10 and top[0][1] == pytest.approx(1.0, abs=1e-5)
        assert top[1][0] == 30

    def test_build_index_returns_queryable(self):
        from backend.retrieval.ann import build_index
        idx = build_index([[1.0, 0.0], [0.0, 1.0]], [0, 1], dim=2)
        assert idx.query([1.0, 0.0], 1)[0][0] == 0

    def test_lexical_reranker_orders_by_relevance(self):
        from backend.retrieval.rerank import LexicalReranker
        docs = [("the weather is sunny in paris today", {"title": "Weather"}),
                ("proximal policy optimization uses a KL penalty", {"title": "PPO"})]
        out = LexicalReranker().rerank("PPO KL penalty optimization", docs, top_n=2)
        assert out[0][1]["title"] == "PPO" and out[0][2] > out[1][2]

    def test_get_reranker_graceful_fallback(self):
        from backend.retrieval.rerank import get_reranker, NoopReranker, LexicalReranker
        assert isinstance(get_reranker("none"), NoopReranker)
        assert isinstance(get_reranker("llm", chat_fn=None), LexicalReranker)  # no chat_fn -> lexical
        assert isinstance(get_reranker("lexical"), LexicalReranker)


# ── Citation grounding ──────────────────────────────────────────────────────
class TestGrounding:
    CITES = [{"num": 1, "title": "PPO", "url": "u1", "source": "arxiv",
              "snippet": "Proximal Policy Optimization uses a KL penalty to constrain the policy update."}]

    def test_supported_claim(self):
        from backend.retrieval.grounding import verify_grounding
        r = verify_grounding("PPO uses a KL penalty to constrain the policy update [1].",
                             self.CITES, threshold=0.3)
        assert r.groundedness == 1.0 and r.n_supported == 1

    def test_fabricated_claim_flagged(self):
        from backend.retrieval.grounding import verify_grounding
        r = verify_grounding("PPO was invented by penguins in Antarctica in 1850 [1].",
                             self.CITES, threshold=0.3)
        assert r.groundedness == 0.0 and len(r.unsupported) == 1

    def test_uncited_claim_detected(self):
        from backend.retrieval.grounding import verify_grounding
        r = verify_grounding("This sentence makes a factual claim with no citation at all.",
                             self.CITES, threshold=0.3)
        assert r.uncited_claims


# ── Query transforms ────────────────────────────────────────────────────────
class TestQueryTransform:
    def test_multi_query_fallback(self):
        from backend.retrieval.query_transform import multi_query
        assert multi_query("what is RLHF", chat_fn=None) == ["what is RLHF"]

    def test_hyde_fallback(self):
        from backend.retrieval.query_transform import hyde
        assert hyde("what is RLHF", chat_fn=None) == ""

    def test_multi_query_with_stub_llm(self):
        from backend.retrieval.query_transform import multi_query
        chat = lambda s, u: '{"queries": ["rlhf reward model", "rlhf ppo"]}'
        out = multi_query("what is RLHF", chat_fn=chat, n=3)
        assert out[0] == "what is RLHF" and "rlhf ppo" in out


# ── Harness gate + ablation ─────────────────────────────────────────────────
class TestHarnessAndAblation:
    def test_gate_passes_offline(self):
        from backend.eval.embedders import LexicalEmbedder
        from backend.eval.judge import GroundednessJudge
        from backend.eval.harness import build_rag, evaluate, DEFAULT_GATE
        rag = build_rag(LexicalEmbedder(), top_k=10, hybrid=True, rerank="lexical")
        agg = evaluate(rag, GroundednessJudge(mode="lexical"))
        for key, threshold in DEFAULT_GATE.items():
            assert agg[key] >= threshold, f"{key}={agg[key]} < {threshold}"

    def test_rerank_does_not_reduce_quality(self):
        from backend.eval.embedders import LexicalEmbedder
        from backend.eval.judge import GroundednessJudge
        from backend.eval.harness import build_rag, evaluate
        emb = LexicalEmbedder(dim=64)  # imperfect dense regime
        j = GroundednessJudge(mode="lexical")
        sem = evaluate(build_rag(emb, top_k=10, hybrid=False, rerank="none"), j)
        full = evaluate(build_rag(emb, top_k=10, hybrid=True, rerank="lexical"), j)
        assert full["mrr"] >= sem["mrr"]
        assert full["ndcg@5"] >= sem["ndcg@5"]

    def test_reflection_ablation_improves_completeness(self):
        from backend.eval.ablation import reflection_ablation
        rows = reflection_ablation()
        assert len(rows) == 2
        draft, reflected = rows
        assert reflected.completeness > draft.completeness
        assert reflected.llm_calls > draft.llm_calls
