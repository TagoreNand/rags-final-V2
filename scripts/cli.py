#!/usr/bin/env python3
"""
scripts/cli.py — RAG Ops management CLI

Usage:
  python scripts/cli.py health
  python scripts/cli.py tasks [--limit N] [--status running]
  python scripts/cli.py task <task_id>
  python scripts/cli.py purge --status failed
  python scripts/cli.py stats
  python scripts/cli.py submit "your research goal" [--steps N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_health(args) -> None:
    import requests
    base = args.base_url.rstrip("/")
    try:
        r = requests.get(f"{base}/health", timeout=15)  # /health probes Ollama+Redis server-side
        h = r.json()
        _icon = lambda ok: "✓" if ok else "✗"
        print(f"  status : {h['status']}")
        print(f"  ollama : {_icon(h['ollama_ok'])}")
        print(f"  redis  : {_icon(h['redis_ok'])}")
        print(f"  db     : {_icon(h['db_ok'])}")
        print(f"  version: {h.get('version', '?')}")
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)


def cmd_tasks(args) -> None:
    import requests
    base = args.base_url.rstrip("/")
    params = {"limit": args.limit}
    hdrs = {"X-API-Key": args.api_key} if args.api_key else {}
    r = requests.get(f"{base}/v1/tasks", params=params, headers=hdrs, timeout=10)
    r.raise_for_status()
    data = r.json()
    tasks = data.get("tasks", [])
    if args.status:
        tasks = [t for t in tasks if t["status"] == args.status]
    if not tasks:
        print("No tasks found.")
        return
    for t in tasks:
        cites = len(t.get("citations", []))
        print(f"  [{t['status']:>9}] {t['task_id'][:8]}…  {t['goal'][:60]}  ({cites} citations)")


def cmd_task(args) -> None:
    import requests
    base = args.base_url.rstrip("/")
    hdrs = {"X-API-Key": args.api_key} if args.api_key else {}
    r = requests.get(f"{base}/v1/tasks/{args.task_id}", headers=hdrs, timeout=10)
    r.raise_for_status()
    t = r.json()
    print(f"Task   : {t['task_id']}")
    print(f"Goal   : {t['goal']}")
    print(f"Status : {t['status']}")
    print(f"Owner  : {t.get('owner', '?')}")
    print(f"Updated: {t['updated_at']}")
    if t.get("error"):
        print(f"Error  : {t['error']}")
    print(f"\nSteps  ({len(t.get('steps', []))}):")
    for s in t.get("steps", []):
        print(f"  [{s['status']:>9}] {s['agent']:12} {s['step_type']}")
    if t.get("citations"):
        print(f"\nCitations ({len(t['citations'])}):")
        for c in t["citations"][:5]:
            print(f"  [{c['num']}] {c['title']} — {c['source']}")
    if t.get("result"):
        print(f"\n--- Result (first 800 chars) ---")
        print(t["result"][:800])


def cmd_purge(args) -> None:
    """Directly purge tasks from the local SQLite DB by status."""
    from backend.db import get_db_paths, connect
    db = get_db_paths()
    with connect(db.db_path) as conn:
        rows = conn.execute(
            "SELECT task_id FROM tasks WHERE status=?", (args.status,)
        ).fetchall()
        if not rows:
            print(f"No tasks with status={args.status}")
            return
        ids = [r["task_id"] for r in rows]
        print(f"Purging {len(ids)} tasks with status={args.status}…")
        for tid in ids:
            conn.execute("DELETE FROM steps WHERE task_id=?", (tid,))
            conn.execute("DELETE FROM docs WHERE task_id=?", (tid,))
            conn.execute("DELETE FROM tasks WHERE task_id=?", (tid,))
        print("Done.")


def cmd_stats(args) -> None:
    """Print DB statistics."""
    from backend.db import get_db_paths, connect
    db = get_db_paths()
    with connect(db.db_path) as conn:
        for status in ("queued", "running", "succeeded", "failed"):
            n = conn.execute("SELECT COUNT(*) FROM tasks WHERE status=?", (status,)).fetchone()[0]
            print(f"  {status:>10}: {n} tasks")
        docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        print(f"  {'docs':>10}: {docs} embedded chunks")
        db_size = db.db_path.stat().st_size / 1024 / 1024
        print(f"  {'db size':>10}: {db_size:.1f} MB")


def cmd_submit(args) -> None:
    import requests
    base = args.base_url.rstrip("/")
    hdrs = {"X-API-Key": args.api_key} if args.api_key else {}
    body = {"goal": args.goal, "max_steps": args.steps, "enable_code_run": False}
    r = requests.post(f"{base}/v1/tasks", json=body, headers=hdrs, timeout=10)
    r.raise_for_status()
    t = r.json()
    print(f"Task created: {t['task_id']}")
    print(f"Status      : {t['status']}")
    print(f"\nPoll: python scripts/cli.py task {t['task_id']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="RAG Ops CLI")
    p.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    p.add_argument("--api-key", default="", help="X-API-Key header value")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="Check API health")

    t = sub.add_parser("tasks", help="List tasks")
    t.add_argument("--limit", type=int, default=20)
    t.add_argument("--status", help="Filter by status")

    td = sub.add_parser("task", help="Show a specific task")
    td.add_argument("task_id")

    pu = sub.add_parser("purge", help="Purge tasks from local DB by status")
    pu.add_argument("--status", required=True, choices=["failed", "queued", "running", "succeeded"])

    sub.add_parser("stats", help="Print DB statistics")

    sm = sub.add_parser("submit", help="Submit a research task")
    sm.add_argument("goal")
    sm.add_argument("--steps", type=int, default=6)

    args = p.parse_args()
    dispatch = {
        "health": cmd_health,
        "tasks":  cmd_tasks,
        "task":   cmd_task,
        "purge":  cmd_purge,
        "stats":  cmd_stats,
        "submit": cmd_submit,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
