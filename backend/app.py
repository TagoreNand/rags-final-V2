"""
backend/app.py

FastAPI application entry-point.
- Mounts versioned /v1 router
- Health check
- Request-ID + CORS middleware
- OTEL + structlog bootstrap
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.agents.pipeline import TaskManager
from backend.api.metrics import metrics_router
from backend.api.v1 import router as v1_router, set_manager
from backend.config import get_settings
from backend.db import get_db_paths, init_db
from backend.models import HealthResponse
from backend.tracing import configure_telemetry, get_logger

log = get_logger(__name__)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
cfg = get_settings()
configure_telemetry(
    cfg.otel_service_name,
    cfg.otel_exporter_otlp_endpoint,
    cfg.otel_enabled,
    cfg.log_level,
    cfg.log_format,
)

DB_PATHS = get_db_paths()
init_db(DB_PATHS.db_path)

_manager = TaskManager(db_paths=DB_PATHS)
set_manager(_manager)  # inject into v1 router

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic RAG Ops",
    version="2.0.0",
    description=(
        "Production-grade multi-source agentic RAG system with "
        "citations, auth, async workers, and full observability."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-ID"] = req_id
    log.info(
        "http",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=elapsed,
        req_id=req_id,
    )
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(v1_router, prefix="/v1")
app.include_router(metrics_router)  # exposes GET /metrics (Prometheus scrape)

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>RAG Ops v2</h1><p><a href='/docs'>API docs</a></p>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    import requests as _req

    ollama_ok = False
    try:
        r = _req.get(cfg.ollama_base_url.rstrip("/") + "/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    redis_ok = False
    try:
        from redis import Redis

        Redis.from_url(cfg.redis_url, socket_connect_timeout=2).ping()
        redis_ok = True
    except Exception:
        pass

    db_ok = False
    try:
        from backend.db import connect

        with connect(DB_PATHS.db_path) as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (ollama_ok and db_ok) else "degraded",
        ollama_ok=ollama_ok,
        redis_ok=redis_ok,
        db_ok=db_ok,
    )
