"""
backend/tools/code_exec.py

Tiered Python code execution:
  Tier 1: subprocess (original, unsafe – dev only)
  Tier 2: Docker container with resource limits + read-only FS (production)
  Tier 3: None / dry-run (returns code without executing)

The sandbox tier is chosen by the CODE_SANDBOX env var.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

from backend.config import get_settings
from backend.tracing import get_logger, timed

log = get_logger(__name__)


@dataclass(frozen=True)
class CodeRunResult:
    exit_code: int
    stdout: str
    stderr: str
    sandbox: str  # which tier was used


def _run_subprocess(code: str, timeout_s: int) -> CodeRunResult:
    """Tier 1 – subprocess, no isolation. Dev only."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        path = f.name
        f.write(code)
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            k.startswith(p) for p in ("API_", "JWT_", "SECRET", "KEY", "TOKEN", "PASS")
        )
    }
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        return CodeRunResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            sandbox="subprocess",
        )
    except subprocess.TimeoutExpired:
        return CodeRunResult(
            exit_code=-1,
            stdout="",
            stderr=f"Timeout after {timeout_s}s",
            sandbox="subprocess",
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_docker(code: str, timeout_s: int) -> CodeRunResult:
    """
    Tier 2 – Docker container with:
      --network none (no internet access from generated code)
      --memory 256m
      --cpus 0.5
      --read-only (except /tmp)
      --user nobody
    Requires Docker daemon accessible.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        path = f.name
        f.write(code)

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--memory",
        "256m",
        "--cpus",
        "0.5",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--user",
        "nobody",
        "--volume",
        f"{path}:/code/script.py:ro",
        "python:3.11-slim",
        "python",
        "/code/script.py",
    ]

    try:
        proc = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s + 10,
        )
        return CodeRunResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            sandbox="docker",
        )
    except subprocess.TimeoutExpired:
        return CodeRunResult(
            exit_code=-1,
            stdout="",
            stderr=f"Docker timeout after {timeout_s}s",
            sandbox="docker",
        )
    except FileNotFoundError:
        log.warning("docker_not_found", fallback="subprocess")
        return _run_subprocess(code, timeout_s)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@timed
def run_code(code: str, timeout_s: Optional[int] = None) -> CodeRunResult:
    cfg = get_settings()
    t = timeout_s or cfg.max_code_run_seconds
    # Re-read sandbox from env each call so test monkeypatching works
    sandbox = os.environ.get("CODE_SANDBOX", cfg.code_sandbox).lower()

    if sandbox == "docker":
        return _run_docker(code, t)
    elif sandbox == "subprocess":
        return _run_subprocess(code, t)
    else:
        # dry-run: return code without executing
        log.info("code_dryrun", sandbox=sandbox)
        return CodeRunResult(
            exit_code=0,
            stdout="[DRY RUN – code not executed]",
            stderr="",
            sandbox="none",
        )
