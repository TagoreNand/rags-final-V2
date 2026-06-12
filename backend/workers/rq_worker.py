"""
backend/workers/rq_worker.py

Redis Queue (RQ) integration.

Architecture:
  - FastAPI route enqueues `run_task_job` into Redis via RQ
  - One or more `rq worker rag-ops` processes pick up and execute jobs
  - Job results / status are mirrored to SQLite for API reads

Run workers:
  rq worker --with-scheduler --url $REDIS_URL rag-ops

Or via the convenience script:
  python -m backend.workers.rq_worker
"""

from __future__ import annotations

import sys

from backend.config import get_settings
from backend.db import get_db_paths, init_db
from backend.tracing import get_logger, configure_telemetry

log = get_logger(__name__)


# ── Job function (executed inside the RQ worker process) ─────────────────────


def run_task_job(task_id: str) -> str:
    """
    Entrypoint called by RQ workers.
    Reconstructs TaskManager and runs the full pipeline.
    Returns task_id on success.
    """
    cfg = get_settings()
    configure_telemetry(
        cfg.otel_service_name,
        cfg.otel_exporter_otlp_endpoint,
        cfg.otel_enabled,
        cfg.log_level,
        cfg.log_format,
    )

    from backend.agents.pipeline import TaskManager  # lazy import in worker process

    db_paths = get_db_paths()
    init_db(db_paths.db_path)

    log.info("worker_start", task_id=task_id)
    manager = TaskManager(db_paths=db_paths)
    manager.run_task(task_id)
    log.info("worker_done", task_id=task_id)
    return task_id


# ── Queue helper (called from FastAPI) ───────────────────────────────────────


def get_queue():
    """Returns the RQ Queue, or None if Redis is unavailable."""
    try:
        from redis import Redis
        from rq import Queue

        cfg = get_settings()
        redis_conn = Redis.from_url(cfg.redis_url, socket_connect_timeout=2)
        redis_conn.ping()
        return Queue("rag-ops", connection=redis_conn, default_timeout=cfg.job_timeout)
    except Exception as exc:
        log.warning("redis_unavailable", error=str(exc))
        return None


def enqueue_task(task_id: str) -> bool:
    """
    Enqueues `run_task_job` into RQ.
    Falls back to a background thread if Redis is unavailable (no-Redis dev mode).
    Returns True if queued remotely, False if running in a thread.
    """
    q = get_queue()
    if q is not None:
        q.enqueue(run_task_job, task_id, job_id=task_id)
        log.info("task_enqueued", task_id=task_id)
        return True

    # Fallback: run in a background thread so the API response returns immediately
    # and the UI can poll for status updates while the task runs.
    import threading

    log.warning("fallback_thread", task_id=task_id)
    thread = threading.Thread(target=run_task_job, args=(task_id,), daemon=True)
    thread.start()
    return False


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Start an RQ worker process."""
    try:
        from redis import Redis
        from rq import Worker
    except ImportError:
        print("ERROR: Install rq and redis: pip install rq redis", file=sys.stderr)
        sys.exit(1)

    cfg = get_settings()
    configure_telemetry(
        cfg.otel_service_name,
        cfg.otel_exporter_otlp_endpoint,
        cfg.otel_enabled,
        cfg.log_level,
        cfg.log_format,
    )

    conn = Redis.from_url(cfg.redis_url)
    worker = Worker(["rag-ops"], connection=conn)
    log.info("rq_worker_starting", queue="rag-ops", redis=cfg.redis_url)
    worker.work(with_scheduler=True)
