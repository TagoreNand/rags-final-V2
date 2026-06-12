"""
backend/api/metrics.py

Prometheus metrics for the RAG Ops API.

Exposes:
  - rag_tasks_total{status}          counter
  - rag_task_duration_seconds        histogram
  - rag_retrieval_docs_total{source} counter
  - rag_llm_latency_seconds          histogram
  - rag_active_tasks                 gauge

Mount in app.py:
    from backend.api.metrics import metrics_router, track_task, track_retrieval, track_llm
    app.include_router(metrics_router)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from fastapi import APIRouter

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    from fastapi.responses import Response as _Response

    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

metrics_router = APIRouter()

if _PROM_AVAILABLE:
    _REGISTRY = CollectorRegistry(auto_describe=True)

    task_counter = Counter(
        "rag_tasks_total",
        "Total tasks by final status",
        ["status"],
        registry=_REGISTRY,
    )
    task_duration = Histogram(
        "rag_task_duration_seconds",
        "End-to-end task duration",
        ["status"],
        registry=_REGISTRY,
        buckets=(5, 15, 30, 60, 120, 300, 600),
    )
    retrieval_counter = Counter(
        "rag_retrieval_docs_total",
        "Retrieved docs by source",
        ["source"],
        registry=_REGISTRY,
    )
    llm_latency = Histogram(
        "rag_llm_latency_seconds",
        "LLM call latency",
        ["model"],
        registry=_REGISTRY,
        buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
    )
    active_tasks = Gauge(
        "rag_active_tasks",
        "Currently running tasks",
        registry=_REGISTRY,
    )

    groundedness = Histogram(
        "rag_groundedness",
        "Citation groundedness of finalized reports (supported cited claims / cited claims)",
        registry=_REGISTRY,
        buckets=(0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
    )

    @metrics_router.get("/metrics", tags=["System"], include_in_schema=False)
    def prometheus_metrics():
        return _Response(
            content=generate_latest(_REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    def track_task(status: str, duration_s: float) -> None:
        task_counter.labels(status=status).inc()
        task_duration.labels(status=status).observe(duration_s)

    def track_retrieval(source: str, count: int = 1) -> None:
        retrieval_counter.labels(source=source).inc(count)

    def track_llm(model: str, latency_s: float) -> None:
        llm_latency.labels(model=model).observe(latency_s)

    def track_groundedness(score: float) -> None:
        groundedness.observe(max(0.0, min(1.0, score)))

    @contextmanager
    def active_task_ctx() -> Generator[None, None, None]:
        active_tasks.inc()
        try:
            yield
        finally:
            active_tasks.dec()

else:
    # Stub implementations when prometheus_client is not installed
    @metrics_router.get("/metrics", tags=["System"], include_in_schema=False)
    def prometheus_metrics():
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("# prometheus_client not installed\n")

    def track_task(status: str, duration_s: float) -> None:
        pass

    def track_retrieval(source: str, count: int = 1) -> None:
        pass

    def track_llm(model: str, latency_s: float) -> None:
        pass

    def track_groundedness(score: float) -> None:
        pass

    @contextmanager
    def active_task_ctx() -> Generator[None, None, None]:
        yield
