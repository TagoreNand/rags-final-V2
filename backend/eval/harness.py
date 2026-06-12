"""
backend/eval/harness.py

Reproducible evaluation harness + CI quality gate.

Ingests the in-repo golden corpus into the REAL LocalRag stack (ANN candidate
generation, BM25 hybrid fusion, reranking, MMR are all exercised) and reports:

  * retrieval metrics  — recall@k, nDCG@k, MRR, MAP
  * faithfulness       — groundedness of the reference answer in retrieved context

Modes:
  python -m backend.eval.harness                 # single enhanced run
  python -m backend.eval.harness --gate          # enforce thresholds (CI)
  python -m backend.eval.harness --compare       # stage-by-stage ablation table
  python -m backend.eval.harness --embedder ollama --judge llm   # production models

Runs fully offline by default (LexicalEmbedder + lexical judge) so the gate is
deterministic and needs no Ollama/GPU.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List

from backend.db import DbPaths, init_db
from backend.eval import golden as G
from backend.eval import metrics as Mx
from backend.eval.embedders import LexicalEmbedder, OllamaEmbedder
from backend.eval.judge import GroundednessJudge
from backend.llm.ollama import OllamaConfig
from backend.retrieval.rag import LocalRag, RagConfig
from backend.tracing import configure_telemetry


def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_rag(embedder, *, top_k: int = 10, hybrid: bool = True, rerank: str = "none") -> LocalRag:
    import backend.retrieval.rag as R

    R.ollama_embeddings = lambda cfg, texts: embedder.embed(texts)
    tmp = Path(tempfile.mkdtemp(prefix="rageval_"))
    db = DbPaths(root_dir=tmp, db_path=tmp / "eval.db")
    init_db(db.db_path)
    cfg = OllamaConfig(base_url="http://x", model="m", embed_model="e")
    rag = LocalRag(db, cfg, RagConfig(top_k=top_k, hybrid_enabled=hybrid, rerank_backend=rerank))
    for d in G.CORPUS:
        rag.ingest_text(task_id="eval", text=d.text, metadata=d.to_meta())
    return rag


def evaluate(rag: LocalRag, judge: GroundednessJudge, *, fusion: bool = False,
             ks=(3, 5, 10)) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    faiths: List[float] = []
    for item in G.GOLDEN:
        if fusion and hasattr(rag, "retrieve_fused"):
            blocks = rag.retrieve_fused([item.question], task_id="eval")
        else:
            blocks = rag.retrieve(item.question, task_id="eval")
        ranked = _dedupe([m.get("url", "") for _t, m in blocks])[: max(ks)]
        row: Dict[str, float] = {}
        for k in ks:
            row[f"recall@{k}"] = Mx.recall_at_k(ranked, item.relevant, k)
            row[f"ndcg@{k}"] = Mx.ndcg_at_k(ranked, item.relevant, k)
        row["mrr"] = Mx.reciprocal_rank(ranked, item.relevant)
        row["map"] = Mx.average_precision(ranked, item.relevant)
        rows.append(row)
        contexts = [G.CORPUS_BY_ID[i].text for i in ranked[:5] if i in G.CORPUS_BY_ID]
        faiths.append(judge.score(item.reference_answer, contexts).faithfulness)
    agg = Mx.aggregate_metrics(rows)
    agg["faithfulness"] = sum(faiths) / len(faiths) if faiths else 0.0
    return agg


DEFAULT_GATE = {"recall@5": 0.80, "ndcg@5": 0.70, "mrr": 0.70, "faithfulness": 0.70}
_COLS = ["recall@3", "recall@5", "recall@10", "ndcg@5", "mrr", "map", "faithfulness"]


def _print_row(label: str, agg: Dict[str, float]) -> None:
    cells = "  ".join(f"{agg.get(c, 0.0):.3f}" for c in _COLS)
    print(f"  {label:<26} {cells}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="RAG Ops evaluation harness")
    p.add_argument("--embedder", choices=["lexical", "ollama"], default="lexical")
    p.add_argument("--judge", choices=["lexical", "embedding", "llm"], default="lexical")
    p.add_argument("--rerank", choices=["none", "lexical", "llm", "cross-encoder"], default="lexical")
    p.add_argument("--no-hybrid", action="store_true")
    p.add_argument("--fusion", action="store_true")
    p.add_argument("--compare", action="store_true", help="stage-by-stage ablation table")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--ablation", action="store_true", help="reflection-loop A/B")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--dim", type=int, default=1024, help="LexicalEmbedder dim (lower = harder dense regime)")
    args = p.parse_args(argv)

    configure_telemetry("rag-ops-eval", "", False, "error", "text")  # quiet logs
    if args.ablation:
        from backend.eval.ablation import reflection_ablation, format_table
        print(format_table(reflection_ablation(GroundednessJudge(mode=args.judge))))
        return 0
    embedder = LexicalEmbedder(dim=args.dim) if args.embedder == "lexical" else OllamaEmbedder(
        OllamaConfig(base_url="http://localhost:11434", model="llama3", embed_model="nomic-embed-text")
    )
    judge = GroundednessJudge(mode=args.judge, embed_fn=embedder.embed)

    print(f"\n  RAG Ops — Retrieval Evaluation   [{len(G.GOLDEN)} golden Q · {len(G.CORPUS)} docs · embedder={args.embedder}]")
    print("  " + "-" * 86)
    print(f"  {'stage':<26} {'  '.join(_COLS)}")
    print("  " + "-" * 86)

    if args.compare:
        stages = [
            ("semantic (ANN only)", dict(hybrid=False, rerank="none"), False),
            ("+ BM25 hybrid (RRF)", dict(hybrid=True, rerank="none"), False),
            ("+ rerank (lexical)", dict(hybrid=True, rerank="lexical"), False),
        ]
        last = {}
        for label, cfg, fusion in stages:
            rag = build_rag(embedder, top_k=args.top_k, **cfg)
            last = evaluate(rag, judge, fusion=fusion)
            _print_row(label, last)
        print("  " + "-" * 86)
        agg = last
    else:
        rag = build_rag(embedder, top_k=args.top_k,
                        hybrid=not args.no_hybrid, rerank=args.rerank)
        agg = evaluate(rag, judge, fusion=args.fusion)
        _print_row("result", agg)
        print("  " + "-" * 86)

    if args.gate:
        failures = [(k, agg.get(k, 0.0), v) for k, v in DEFAULT_GATE.items() if agg.get(k, 0.0) < v]
        if failures:
            print("  GATE: FAIL")
            for k, got, need in failures:
                print(f"    {k}: {got:.3f} < {need:.2f}")
            return 1
        print(f"  GATE: PASS  (recall@5>={DEFAULT_GATE['recall@5']}, ndcg@5>={DEFAULT_GATE['ndcg@5']}, "
              f"mrr>={DEFAULT_GATE['mrr']}, faithfulness>={DEFAULT_GATE['faithfulness']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
