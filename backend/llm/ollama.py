"""
backend/llm/ollama.py
Resilient Ollama API client with:
  - Exponential back-off retries (tenacity)
  - Per-call timeout
  - Structured error types
  - Token usage tracking
  - Embedding batching
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.tracing import get_logger, timed

log = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM backend returns an unrecoverable error."""


class LLMTimeoutError(LLMError):
    pass


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    embed_model: str
    timeout_s: int = 120
    max_retries: int = 3


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


def _make_session(max_retries: int) -> requests.Session:
    retry = Retry(
        total=max_retries,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _session(cfg: OllamaConfig) -> requests.Session:
    # Create one session per config hash (thread-safe since GIL protects dict)
    key = (cfg.base_url, cfg.max_retries)
    if key not in _SESSIONS:
        _SESSIONS[key] = _make_session(cfg.max_retries)
    return _SESSIONS[key]


_SESSIONS: Dict[tuple, requests.Session] = {}


@timed
def ollama_chat(
    cfg: OllamaConfig,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    format_json: bool = False,
    system_override: Optional[str] = None,
) -> LLMResponse:
    """
    Calls /api/chat and returns a LLMResponse.
    `format_json=True` appends a JSON-output hint to the system prompt.
    """
    url = cfg.base_url.rstrip("/") + "/api/chat"
    prompt_messages = list(messages)

    if system_override:
        prompt_messages = [{"role": "system", "content": system_override}] + [
            m for m in prompt_messages if m.get("role") != "system"
        ]

    if format_json:
        hint = (
            "\n\nIMPORTANT: Return ONLY valid JSON. "
            "No markdown fences, no prose. All string values must be valid JSON strings."
        )
        if prompt_messages and prompt_messages[0]["role"] == "system":
            prompt_messages[0] = dict(prompt_messages[0])
            prompt_messages[0]["content"] += hint
        else:
            prompt_messages.insert(0, {"role": "system", "content": hint})

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": prompt_messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if format_json:
        payload["format"] = "json"

    t0 = time.perf_counter()
    try:
        resp = _session(cfg).post(url, json=payload, timeout=cfg.timeout_s)
        resp.raise_for_status()
    except requests.Timeout as exc:
        raise LLMTimeoutError(f"Ollama timed out after {cfg.timeout_s}s") from exc
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc

    data = resp.json()
    content = data["message"]["content"]
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    usage = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
    log.debug(
        "llm_call",
        model=cfg.model,
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        latency_ms=latency_ms,
    )

    return LLMResponse(
        content=content,
        model=cfg.model,
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        latency_ms=latency_ms,
    )


@timed
def ollama_embeddings(cfg: OllamaConfig, texts: List[str]) -> List[List[float]]:
    """
    Embed texts via Ollama /api/embed (batch) with single-item fallback.
    """
    url = cfg.base_url.rstrip("/") + "/api/embed"
    sess = _session(cfg)

    # Try batch endpoint first (Ollama ≥ 0.1.31)
    try:
        resp = sess.post(
            url, json={"model": cfg.embed_model, "input": texts}, timeout=cfg.timeout_s
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data.get("embeddings"), list):
            return data["embeddings"]
    except Exception:
        pass

    # Legacy: /api/embeddings one at a time
    url_legacy = cfg.base_url.rstrip("/") + "/api/embeddings"
    out: List[List[float]] = []
    for t in texts:
        resp = sess.post(
            url_legacy,
            json={"model": cfg.embed_model, "prompt": t},
            timeout=cfg.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if "embedding" not in data:
            raise LLMError(f"No embedding in response: {json.dumps(data)[:200]}")
        out.append(data["embedding"])
    return out
