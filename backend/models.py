"""
backend/models.py
Pydantic v2 request / response models.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

TaskStatus = Literal["queued", "running", "succeeded", "failed"]
StepStatus = Literal["queued", "running", "succeeded", "failed"]


# ── Requests ──────────────────────────────────────────────────────────────────


class TaskCreateRequest(BaseModel):
    goal: str = Field(
        ..., min_length=5, max_length=2000, description="Research goal / question"
    )
    max_steps: int = Field(8, ge=1, le=25)
    enable_code_run: bool = Field(
        False, description="Allow code execution (requires code_sandbox=docker in prod)"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Restrict retrieval to specific sources: 'wikipedia', 'arxiv', 'brave'",
    )


class TokenRequest(BaseModel):
    api_key: str


# ── Responses ─────────────────────────────────────────────────────────────────


class CitationOut(BaseModel):
    num: int
    title: str
    url: str
    source: str
    snippet: str


class StepOut(BaseModel):
    step_id: str
    agent: str
    step_type: str
    status: StepStatus
    output: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class TaskResponse(BaseModel):
    task_id: str
    goal: str
    status: TaskStatus
    owner: str = "anonymous"
    created_at: str
    updated_at: str
    error: Optional[str] = None
    steps: List[StepOut] = Field(default_factory=list)
    result: Optional[str] = None
    citations: List[CitationOut] = Field(default_factory=list)


class TaskListResponse(BaseModel):
    tasks: List[TaskResponse]
    total: int


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class HealthResponse(BaseModel):
    status: str
    ollama_ok: bool
    redis_ok: bool
    db_ok: bool
    version: str = "2.0.0"
