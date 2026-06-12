"""
backend/agents/pipeline.py

Upgraded agent pipeline:
  - Multi-source retrieval (Wikipedia + arXiv + Brave)
  - MMR diversity in RAG
  - Citation tracking and bibliography generation
  - Structured logging + OTEL spans per step
  - Retry logic per LLM call
  - Graceful partial results on failure
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from backend.config import get_settings
from backend.db import (
    DbPaths,
    get_task,
    insert_step,
    list_task_steps,
    set_task_status,
    update_step,
    upsert_task,
)
from backend.llm.ollama import OllamaConfig, LLMError, LLMTimeoutError, ollama_chat
from backend.models import TaskResponse, StepOut, CitationOut
from backend.retrieval.citations import CitationRegistry, format_context_with_citations
from backend.retrieval.grounding import verify_grounding
from backend.retrieval.query_transform import hyde
from backend.retrieval.rag import LocalRag, RagConfig
from backend.retrieval.sources import multi_source_search
from backend.tools.code_exec import run_code
from backend.tracing import get_logger, span, timed

log = get_logger(__name__)


def _metrics():
    """
    Lazily import the Prometheus metric hooks.

    Imported lazily (not at module top) because ``backend.api`` eagerly imports
    the v1 router, which imports this module — a top-level import here would
    create a circular import at load time. By the time a task actually runs both
    modules are fully initialised, so this resolves cleanly and is cached in
    ``sys.modules``. Returns ``None`` when metrics are unavailable, keeping every
    call site best-effort so instrumentation can never break the pipeline.
    """
    try:
        from backend.api import metrics as _m

        return _m
    except Exception:  # pragma: no cover - metrics are strictly best-effort
        return None


# ── JSON helpers ──────────────────────────────────────────────────────────────


def _safe_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    for attempt in (s, s[s.find("{") : s.rfind("}") + 1]):
        try:
            return json.loads(attempt)
        except Exception:
            pass
    return None


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        if t.endswith("```"):
            t = t[:-3].strip()
    return t


# ── Agent settings ────────────────────────────────────────────────────────────


class TaskManager:
    def __init__(self, db_paths: DbPaths):
        self.db_paths = db_paths
        cfg = get_settings()

        self.ollama_cfg = OllamaConfig(
            base_url=cfg.ollama_base_url,
            model=cfg.llm_model,
            embed_model=cfg.embed_model,
            max_retries=cfg.http_max_retries,
        )
        self.rag = LocalRag(
            db_paths=db_paths,
            ollama_cfg=self.ollama_cfg,
            rag_cfg=RagConfig(
                top_k=cfg.rag_top_k,
                chunk_size=cfg.rag_chunk_size,
                chunk_overlap=cfg.rag_chunk_overlap,
                rerank_backend=cfg.rerank_backend,
                rrf_k=cfg.rrf_k,
                hybrid_enabled=cfg.hybrid_enabled,
                ann_enabled=cfg.ann_enabled,
            ),
        )
        self._task_opts: Dict[str, Dict[str, Any]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def create_task(
        self,
        goal: str,
        max_steps: int,
        enable_code_run: bool,
        owner: str = "anonymous",
        sources: Optional[List[str]] = None,
    ) -> TaskResponse:
        task_id = str(uuid.uuid4())
        upsert_task(self.db_paths.db_path, task_id, goal, status="queued", owner=owner)
        self._task_opts[task_id] = {
            "max_steps": max_steps,
            "enable_code_run": enable_code_run,
            "sources": sources or [],
        }
        return self._build_response(task_id)

    def get_task_response(self, task_id: str) -> Optional[TaskResponse]:
        return self._build_response(task_id)

    # ── Main pipeline ──────────────────────────────────────────────────────

    @timed
    def run_task(self, task_id: str) -> None:
        from contextlib import nullcontext

        m = _metrics()
        t = get_task(self.db_paths.db_path, task_id)
        if not t:
            return

        goal: str = t["goal"]
        opts = self._task_opts.get(task_id, {})
        cfg = get_settings()
        max_steps = int(opts.get("max_steps", cfg.max_steps))
        enable_code_run = bool(opts.get("enable_code_run", cfg.enable_code_run))
        allowed_sources: List[str] = opts.get("sources", [])

        set_task_status(self.db_paths.db_path, task_id, "running")
        citations = CitationRegistry()
        step_count = 0
        report_markdown: Optional[str] = None
        started_at = time.perf_counter()
        active_ctx = m.active_task_ctx() if m else nullcontext()

        try:
            with active_ctx, span("pipeline", {"task_id": task_id, "goal": goal[:80]}):
                # ── 1. Supervisor: plan queries ──────────────────────────
                step_count += 1
                sup_step = self._step(task_id, "supervisor", "web_research")
                plan = self._supervisor_plan(goal) or {
                    "queries": [goal],
                    "include_code_demo": False,
                }
                queries = [
                    str(q).strip()
                    for q in plan.get("queries", [goal])
                    if str(q).strip()
                ][: cfg.rag_max_sources]
                include_code = bool(plan.get("include_code_demo", False))
                self._ok(
                    sup_step,
                    json.dumps(
                        {"queries": queries, "include_code_demo": include_code},
                        indent=2,
                    ),
                )

                # ── 2. Researcher: multi-source retrieval + ingest ───────
                if step_count >= max_steps:
                    raise RuntimeError("Max steps reached before research.")
                step_count += 1
                res_step = self._step(task_id, "researcher", "web_research")
                ingested_meta: List[Dict[str, Any]] = []

                for q in queries:
                    docs = multi_source_search(q, max_docs=cfg.rag_max_sources)
                    for doc in docs:
                        if allowed_sources and doc.source not in allowed_sources:
                            continue
                        self.rag.ingest_text(
                            task_id=task_id, text=doc.text, metadata=doc.to_meta()
                        )
                        ingested_meta.append(doc.to_meta())
                        if m:
                            m.track_retrieval(doc.source)
                        if len(ingested_meta) >= cfg.rag_max_sources:
                            break
                    if len(ingested_meta) >= cfg.rag_max_sources:
                        break

                self._ok(
                    res_step,
                    json.dumps({"sources_ingested": ingested_meta}, indent=2)[:4000],
                )

                # ── 3. RAG: retrieve + citation-annotated context ─────────
                if step_count >= max_steps:
                    raise RuntimeError("Max steps reached before RAG.")
                step_count += 1
                rag_step = self._step(task_id, "rager", "rag_index")
                # Query transformation: reuse the supervisor's query variants
                # (no extra LLM calls) and optionally add a HyDE passage, then
                # run RRF-fused retrieval with second-stage reranking.
                variants = [goal] + [q for q in queries if q and q != goal]
                if cfg.hyde_enabled:
                    hp = hyde(goal, chat_fn=self._chat_text)
                    if hp:
                        variants.append(hp)
                if cfg.multi_query_enabled and len(variants) > 1:
                    blocks = self.rag.retrieve_fused(
                        variants, task_id=task_id,
                        rerank=cfg.rerank_backend, chat_fn=self._chat_text,
                    )
                else:
                    blocks = self.rag.retrieve(
                        goal, task_id=task_id,
                        rerank=cfg.rerank_backend, chat_fn=self._chat_text,
                    )
                context = format_context_with_citations(blocks, citations)
                self._ok(
                    rag_step, context[:4000] if context else "No RAG context retrieved."
                )

                # ── 4. Synthesis with reflection loop ─────────────────────
                if step_count >= max_steps:
                    raise RuntimeError("Max steps reached before synthesis.")
                step_count += 1
                synth_step = self._step(task_id, "analyst", "synthesize_report")

                revision_notes: Optional[str] = None
                sources_used: List[str] = []
                for retry in range(3):
                    payload = self._synthesize(goal, context, revision_notes)
                    if isinstance(payload, dict):
                        report_markdown = payload.get("report_markdown") or payload.get(
                            "report", ""
                        )
                        sources_used = payload.get("sources_used") or []
                    else:
                        report_markdown = str(payload)

                    reflect = self._reflect(goal, report_markdown or "")
                    if not reflect or not reflect.get("needs_revision"):
                        break
                    revision_notes = reflect.get("revision_instructions", "")

                if not report_markdown:
                    report_markdown = f"Unable to synthesize a report for: {goal}"

                self._ok(
                    synth_step,
                    json.dumps({"sources_used": sources_used}, indent=2)[:2000]
                    + "\n\n"
                    + (report_markdown[:3500]),
                )

                # ── 5. Optional code demo ─────────────────────────────────
                if include_code and step_count < max_steps:
                    step_count += 1
                    code_step = self._step(task_id, "coder", "code_demo")
                    try:
                        code_text = self._code_demo(goal)
                        if enable_code_run:
                            result = run_code(
                                code_text, timeout_s=cfg.max_code_run_seconds
                            )
                            out = (
                                f"--- CODE ({result.sandbox}) ---\n{code_text[:2500]}\n\n"
                                f"--- STDOUT ---\n{result.stdout[-1500:]}\n"
                                f"--- STDERR ---\n{result.stderr[-500:]}\n"
                                f"exit={result.exit_code}"
                            )
                        else:
                            out = f"--- CODE (not executed) ---\n{code_text[:3500]}"
                        self._ok(code_step, out[:4000])
                    except Exception as exc:
                        self._fail(code_step, str(exc))

                # ── 6. Append bibliography ────────────────────────────────
                bib = citations.bibliography_markdown()
                final_report = (report_markdown or "") + "\n\n" + bib
                citations_json = json.dumps(
                    citations.bibliography_json(), ensure_ascii=False
                )

                grounding_note = ""
                if cfg.grounding_enabled:
                    try:
                        gr = verify_grounding(
                            report_markdown or "",
                            citations.bibliography_json(),
                            threshold=cfg.grounding_threshold,
                        )
                        if m:
                            m.track_groundedness(gr.groundedness)
                        log.info("grounding", task_id=task_id, **gr.to_dict())
                        grounding_note = (
                            f"<!-- groundedness={gr.groundedness:.2f} "
                            f"supported={gr.n_supported}/{gr.n_cited_claims} "
                            f"unsupported={len(gr.unsupported)} "
                            f"uncited={len(gr.uncited_claims)} -->\n"
                        )
                    except Exception as exc:  # pragma: no cover
                        log.warning("grounding_error", error=str(exc))

                fin_step = self._step(task_id, "reflector", "finalize")
                self._ok(fin_step, (grounding_note + final_report)[:8000])
                set_task_status(
                    self.db_paths.db_path,
                    task_id,
                    "succeeded",
                    citations=citations_json,
                )
                if m:
                    m.track_task("succeeded", time.perf_counter() - started_at)
                log.info("task_done", task_id=task_id, n_citations=len(citations.all()))

        except Exception as exc:
            if m:
                m.track_task("failed", time.perf_counter() - started_at)
            log.error("task_failed", task_id=task_id, error=str(exc))
            set_task_status(self.db_paths.db_path, task_id, "failed", error=str(exc))

    # ── LLM helpers ────────────────────────────────────────────────────────

    def _chat_json(
        self, system: str, user: str, temperature: float = 0.2
    ) -> Optional[Dict[str, Any]]:
        try:
            resp = ollama_chat(
                self.ollama_cfg,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                format_json=True,
            )
            self._track_llm(resp)
            return _safe_json(resp.content)
        except (LLMError, LLMTimeoutError) as exc:
            log.warning("llm_error", error=str(exc))
            return None

    def _chat_text(self, system: str, user: str, temperature: float = 0.2) -> str:
        try:
            resp = ollama_chat(
                self.ollama_cfg,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            self._track_llm(resp)
            return resp.content
        except (LLMError, LLMTimeoutError) as exc:
            log.warning("llm_error", error=str(exc))
            return ""

    def _track_llm(self, resp: Any) -> None:
        """Record LLM call latency to Prometheus (best-effort, never raises)."""
        m = _metrics()
        if m:
            try:
                m.track_llm(self.ollama_cfg.model, resp.latency_ms / 1000.0)
            except Exception:  # pragma: no cover - instrumentation must not fail calls
                pass

    def _supervisor_plan(self, goal: str) -> Optional[Dict[str, Any]]:
        system = (
            "You are a supervisor for an agentic multi-source research system. "
            "Plan Wikipedia, academic, and web searches. "
            'Return ONLY valid JSON: {"queries": [<5 short strings>], "include_code_demo": <bool>}.'
        )
        return self._chat_json(system, f"Goal: {goal}\n\nProduce JSON now.")

    def _synthesize(
        self, goal: str, context: str, revision_notes: Optional[str]
    ) -> Any:
        system = (
            "You are an expert analyst. Write a structured research report using ONLY the provided context. "
            "Use [N] citation markers (e.g. [1], [2]) whenever you reference a source. "
            'Return ONLY valid JSON: {"report_markdown": "<markdown>", "sources_used": [<urls>]}.'
        )
        revision = (
            f"\nRevision instructions: {revision_notes}\n" if revision_notes else ""
        )
        user = f"Goal: {goal}{revision}\n\nContext:\n{context}\n\nWrite the report."
        result = self._chat_json(system, user, temperature=0.3)
        return result or self._chat_text(system, user, temperature=0.3)

    def _reflect(self, goal: str, report: str) -> Optional[Dict[str, Any]]:
        system = (
            "You are a strict quality reviewer. Evaluate the research report against the goal. "
            'Return ONLY valid JSON: {"needs_revision": <bool>, "revision_instructions": "<string>"}.'
        )
        rubric = (
            "Criteria: (1) Clear intro, (2) Key findings with source citations [N], "
            "(3) Evidence-backed claims, (4) Actionable conclusion, (5) Acknowledged limitations."
        )
        return self._chat_json(
            system, f"Goal: {goal}\n\nRubric:\n{rubric}\n\nReport:\n{report}"
        )

    def _code_demo(self, goal: str) -> str:
        system = (
            "Write a self-contained Python 3 script demonstrating local RAG retrieval. "
            "Output ONLY the python code, no markdown fences."
        )
        return _strip_fences(self._chat_text(system, f"Goal: {goal}"))

    # ── DB helpers ─────────────────────────────────────────────────────────

    def _step(self, task_id: str, agent: str, step_type: str) -> str:
        step_id = str(uuid.uuid4())
        insert_step(
            self.db_paths.db_path, step_id, task_id, agent, step_type, status="running"
        )
        return step_id

    def _ok(self, step_id: str, output: str) -> None:
        update_step(self.db_paths.db_path, step_id, status="succeeded", output=output)

    def _fail(self, step_id: str, error: str) -> None:
        update_step(self.db_paths.db_path, step_id, status="failed", error=error)

    def _build_response(self, task_id: str) -> Optional[TaskResponse]:
        t = get_task(self.db_paths.db_path, task_id)
        if not t:
            return None
        steps_rows = list_task_steps(self.db_paths.db_path, task_id)
        steps_out = [
            StepOut(
                step_id=row["step_id"],
                agent=row["agent"],
                step_type=row["step_type"],
                status=row["status"],
                output=(row["output"] or "")[:500],
                error=row["error"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in steps_rows
        ]
        citations_json = t["citations"] if "citations" in t.keys() else None
        cite_list: List[CitationOut] = []
        if citations_json:
            try:
                for c in json.loads(citations_json):
                    cite_list.append(
                        CitationOut(
                            **{
                                k: c[k]
                                for k in ("num", "title", "url", "source", "snippet")
                                if k in c
                            }
                        )
                    )
            except Exception:
                pass

        result = next((s.output for s in steps_out if s.step_type == "finalize"), None)
        return TaskResponse(
            task_id=t["task_id"],
            goal=t["goal"],
            status=t["status"],
            owner=t["owner"] if "owner" in t.keys() else "anonymous",
            created_at=t["created_at"],
            updated_at=t["updated_at"],
            error=t["error"],
            steps=steps_out,
            result=result,
            citations=cite_list,
        )
