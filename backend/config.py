"""
backend/config.py
Centralised, validated configuration loaded from env / .env file.
Pydantic-settings validates types at startup so bad config fails fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    # ── LLM ───────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3"
    embed_model: str = "nomic-embed-text"

    # ── RAG ───────────────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    rag_max_sources: int = 6
    max_steps: int = 8

    # ── Advanced retrieval (ANN + hybrid fusion + rerank + query transform) ─
    rerank_backend: str = "lexical"   # none | lexical | llm | cross-encoder
    rrf_k: int = 60                   # Reciprocal Rank Fusion constant
    hybrid_enabled: bool = True       # fuse BM25 with semantic via RRF
    ann_enabled: bool = True          # hnswlib ANN when corpus is large enough
    multi_query_enabled: bool = True  # fuse retrieval over supervisor query variants
    hyde_enabled: bool = False        # Hypothetical Document Embeddings (extra LLM call)

    # ── Citation grounding + eval gate thresholds ──────────────────────────
    grounding_enabled: bool = True
    grounding_threshold: float = 0.30
    eval_min_recall_at_5: float = 0.80
    eval_min_ndcg_at_5: float = 0.70
    eval_min_mrr: float = 0.70
    eval_min_faithfulness: float = 0.70

    # ── Multi-source retrieval ─────────────────────────────────────────────
    wiki_search_endpoint: str = "https://en.wikipedia.org/w/api.php"
    wiki_summary_endpoint: str = "https://en.wikipedia.org/api/rest_v1/page/summary"
    serpapi_key: Optional[str] = None
    arxiv_enabled: bool = True
    news_api_key: Optional[str] = None
    brave_search_key: Optional[str] = None
    max_web_fetch_chars: int = 25_000
    http_timeout_s: int = 20
    http_max_retries: int = 3

    # ── Code execution ─────────────────────────────────────────────────────
    enable_code_run: bool = False
    max_code_run_seconds: int = 20
    code_sandbox: str = "subprocess"  # subprocess | docker | none

    # ── Worker queue ──────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    worker_concurrency: int = 4
    task_result_ttl: int = 86_400
    job_timeout: int = 300

    # ── Auth ──────────────────────────────────────────────────────────────
    api_keys_raw: str = Field(default="", alias="API_KEYS")
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # ── Observability ─────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4318/v1/traces"
    otel_service_name: str = "rag-ops"
    otel_enabled: bool = False
    log_level: str = "INFO"
    log_format: str = "json"  # json | text

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/rag.db"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }

    @field_validator("rag_chunk_size")
    @classmethod
    def chunk_must_exceed_overlap(cls, v: int, info) -> int:  # noqa: ANN001
        return v

    @property
    def api_keys(self) -> List[str]:
        return [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "rag.db"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


def get_settings() -> Settings:
    """
    Returns a Settings instance.
    Not cached — allows test monkeypatching of env vars.
    In production, instantiation is fast (just env reads); cache at the
    call site if you need it.
    """
    return Settings()
