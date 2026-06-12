"""
backend/api/v1.py

Versioned API router — mounted at /v1 by app.py.
Keeping routes in their own module makes it trivial to add /v2 later
and keeps app.py clean (just startup + middleware).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response


from backend.agents.pipeline import TaskManager
from backend.auth.middleware import create_access_token, require_auth
from backend.config import get_settings
from backend.db import get_db_paths, list_tasks, set_task_status
from backend.models import (
    TaskCreateRequest,
    TaskListResponse,
    TaskResponse,
    TokenRequest,
    TokenResponse,
)
from backend.tracing import get_logger
from backend.workers.rq_worker import enqueue_task

log = get_logger(__name__)

router = APIRouter()

# Shared TaskManager — injected by app.py at startup via router.state
_manager: Optional[TaskManager] = None


def set_manager(m: TaskManager) -> None:
    global _manager
    _manager = m


def get_manager() -> TaskManager:
    if _manager is None:
        raise RuntimeError("TaskManager not initialised.")
    return _manager


# ── Auth ──────────────────────────────────────────────────────────────────────


@router.post(
    "/auth/token",
    response_model=TokenResponse,
    tags=["Auth"],
    summary="Exchange an API key for a short-lived JWT bearer token",
)
def issue_token(req: TokenRequest) -> TokenResponse:
    cfg = get_settings()
    if not cfg.auth_enabled or req.api_key not in cfg.api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    token = create_access_token(sub=req.api_key)
    return TokenResponse(access_token=token, expires_in=cfg.jwt_expire_minutes * 60)


# ── Tasks ─────────────────────────────────────────────────────────────────────


@router.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=202,
    tags=["Tasks"],
    summary="Create and enqueue a new research task",
)
def create_task(
    req: TaskCreateRequest,
    identity: str = Depends(require_auth),
) -> TaskResponse:
    mgr = get_manager()
    task = mgr.create_task(
        goal=req.goal,
        max_steps=req.max_steps,
        enable_code_run=req.enable_code_run,
        owner=identity,
        sources=req.sources,
    )
    enqueue_task(task.task_id)
    log.info("task_created", task_id=task.task_id, owner=identity)
    return mgr.get_task_response(task.task_id)


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    tags=["Tasks"],
    summary="List recent tasks (scoped to the authenticated user)",
)
def list_all_tasks(
    identity: str = Depends(require_auth),
    limit: int = Query(20, ge=1, le=100),
) -> TaskListResponse:
    mgr = get_manager()
    db = get_db_paths()
    owner = None if identity == "anonymous" else identity
    rows = list_tasks(db.db_path, owner=owner, limit=limit)
    tasks = [mgr.get_task_response(row["task_id"]) for row in rows]
    tasks = [t for t in tasks if t is not None]
    return TaskListResponse(tasks=tasks, total=len(tasks))


@router.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    tags=["Tasks"],
    summary="Poll task status and retrieve results",
)
def get_task(
    task_id: str,
    _: str = Depends(require_auth),
) -> TaskResponse:
    task = get_manager().get_task_response(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@router.delete(
    "/tasks/{task_id}",
    status_code=204,
    tags=["Tasks"],
    summary="Cancel / soft-delete a task",
    response_class=Response,
)
def delete_task(
    task_id: str,
    identity: str = Depends(require_auth),
) -> Response:
    mgr = get_manager()
    task = mgr.get_task_response(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    db = get_db_paths()
    set_task_status(db.db_path, task_id, "failed", error=f"Cancelled by {identity}.")
    return Response(status_code=204)


@router.get(
    "/tasks/{task_id}/citations",
    tags=["Tasks"],
    summary="Retrieve just the citation bibliography for a completed task",
)
def get_citations(
    task_id: str,
    _: str = Depends(require_auth),
) -> dict:
    task = get_manager().get_task_response(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"task_id": task_id, "citations": [c.model_dump() for c in task.citations]}
