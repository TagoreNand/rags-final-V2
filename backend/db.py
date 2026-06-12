"""
backend/db.py

SQLite data layer with:
  - WAL mode + busy timeout for concurrent access
  - Schema versioning / migrations
  - Proper context management
  - Type-safe row helpers
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from backend.tracing import get_logger

log = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DbPaths:
    root_dir: Path
    db_path: Path


def get_db_paths(root: Optional[Path] = None) -> DbPaths:
    if root is None:
        root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return DbPaths(root_dir=root, db_path=data_dir / "rag.db")


# ── Connection ────────────────────────────────────────────────────────────────


@contextmanager
def connect(db_path: Path):
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Schema & migrations ───────────────────────────────────────────────────────

_SCHEMA_VERSION = 2

_MIGRATIONS: Dict[int, str] = {
    1: """
        CREATE TABLE IF NOT EXISTS tasks (
          task_id    TEXT PRIMARY KEY,
          goal       TEXT NOT NULL,
          status     TEXT NOT NULL,
          owner      TEXT DEFAULT 'anonymous',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          error      TEXT
        );|
        CREATE TABLE IF NOT EXISTS steps (
          step_id    TEXT PRIMARY KEY,
          task_id    TEXT NOT NULL REFERENCES tasks(task_id),
          agent      TEXT NOT NULL,
          step_type  TEXT NOT NULL,
          status     TEXT NOT NULL,
          output     TEXT,
          error      TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );|
        CREATE TABLE IF NOT EXISTS docs (
          doc_id    INTEGER PRIMARY KEY AUTOINCREMENT,
          task_id   TEXT,
          text      TEXT NOT NULL,
          metadata  TEXT,
          embedding BLOB NOT NULL
        );|
        CREATE INDEX IF NOT EXISTS idx_steps_task ON steps(task_id);|
        CREATE INDEX IF NOT EXISTS idx_docs_task  ON docs(task_id);
    """,
    2: """
        ALTER TABLE tasks ADD COLUMN citations TEXT;
    """,
}


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)"
        )
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = row[0] or 0

        for version in sorted(_MIGRATIONS):
            if version > current:
                log.info("db_migration", version=version)
                # SQLite doesn't support ALTER TABLE IF NOT EXISTS, so handle gracefully
                statements = [
                    s.strip() for s in _MIGRATIONS[version].split("|") if s.strip()
                ]
                for stmt in statements:
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError as e:
                        if (
                            "duplicate column" in str(e).lower()
                            or "already exists" in str(e).lower()
                        ):
                            pass  # idempotent
                        else:
                            raise
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version VALUES (?)", (version,)
                )


# ── Task CRUD ─────────────────────────────────────────────────────────────────


def upsert_task(
    db_path: Path,
    task_id: str,
    goal: str,
    status: str,
    owner: str = "anonymous",
    error: Optional[str] = None,
) -> None:
    now = utc_now()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(task_id, goal, status, owner, created_at, updated_at, error)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(task_id) DO UPDATE SET
              goal=excluded.goal, status=excluded.status,
              updated_at=excluded.updated_at, error=excluded.error
            """,
            (task_id, goal, status, owner, now, now, error),
        )


def set_task_status(
    db_path: Path,
    task_id: str,
    status: str,
    error: Optional[str] = None,
    citations: Optional[str] = None,
) -> None:
    now = utc_now()
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET status=?, updated_at=?, error=?, citations=? WHERE task_id=?",
            (status, now, error, citations, task_id),
        )


def get_task(db_path: Path, task_id: str) -> Optional[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()


def list_tasks(
    db_path: Path, owner: Optional[str] = None, limit: int = 50
) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        if owner:
            return conn.execute(
                "SELECT * FROM tasks WHERE owner=? ORDER BY created_at DESC LIMIT ?",
                (owner, limit),
            ).fetchall()
        return conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()


# ── Step CRUD ─────────────────────────────────────────────────────────────────


def insert_step(
    db_path: Path, step_id: str, task_id: str, agent: str, step_type: str, status: str
) -> None:
    now = utc_now()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO steps(step_id,task_id,agent,step_type,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            (step_id, task_id, agent, step_type, status, now, now),
        )


def update_step(
    db_path: Path,
    step_id: str,
    status: str,
    output: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    now = utc_now()
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE steps SET status=?, updated_at=?, output=COALESCE(?,output), error=? WHERE step_id=?",
            (status, now, output, error, step_id),
        )


def list_task_steps(db_path: Path, task_id: str) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM steps WHERE task_id=? ORDER BY created_at ASC", (task_id,)
        ).fetchall()


# ── Document / embedding CRUD ────────────────────────────────────────────────


def insert_docs(
    db_path: Path,
    task_id: Optional[str],
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO docs(task_id, text, metadata, embedding) VALUES(?,?,?,?)",
            [
                (
                    task_id,
                    text,
                    json.dumps(meta, ensure_ascii=False),
                    np.array(emb, dtype=np.float32).tobytes(),
                )
                for text, meta, emb in zip(texts, metadatas, embeddings)
            ],
        )


def iter_docs(
    db_path: Path, task_id: Optional[str] = None
) -> Iterable[Tuple[str, Dict[str, Any], str, np.ndarray]]:
    with connect(db_path) as conn:
        sql = "SELECT doc_id, text, metadata, embedding FROM docs"
        params: tuple = ()
        if task_id:
            sql += " WHERE task_id=?"
            params = (task_id,)
        for row in conn.execute(sql, params).fetchall():
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            blob = row["embedding"]
            dim = len(blob) // 4
            vec = np.frombuffer(blob, dtype=np.float32).reshape((dim,))
            yield (str(row["doc_id"]), meta, row["text"], vec)
