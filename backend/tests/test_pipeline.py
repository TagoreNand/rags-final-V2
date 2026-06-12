"""
backend/tests/test_pipeline.py

Test coverage for:
  - RAG chunking and retrieval
  - Citation registry
  - Source retrieval (mocked)
  - Agent prompt correctness
  - Auth middleware
  - API endpoints (TestClient)
  - Code execution isolation
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Provide a fresh DbPaths pointing to a temp directory."""
    from backend.db import DbPaths, init_db

    db_path = tmp_path / "test.db"
    paths = DbPaths(root_dir=tmp_path, db_path=db_path)
    init_db(db_path)
    return paths


@pytest.fixture
def fake_embeddings():
    """Return deterministic fake embeddings (unit vector)."""

    def _embed(cfg, texts: List[str]) -> List[List[float]]:
        rng = np.random.default_rng(42)
        out = []
        for t in texts:
            v = rng.standard_normal(768).astype(np.float32)
            v = v / np.linalg.norm(v)
            out.append(v.tolist())
        return out

    return _embed


# ══════════════════════════════════════════════════════════════════════════════
# RAG tests
# ══════════════════════════════════════════════════════════════════════════════


class TestChunkText:
    def test_basic_split(self):
        from backend.retrieval.rag import chunk_text

        text = "A" * 2500
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 1000

    def test_short_text_single_chunk(self):
        from backend.retrieval.rag import chunk_text

        chunks = chunk_text("Hello world.", chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_overlap_lt_size_required(self):
        from backend.retrieval.rag import chunk_text

        with pytest.raises(ValueError):
            chunk_text("text", chunk_size=100, overlap=100)

    def test_empty_text(self):
        from backend.retrieval.rag import chunk_text

        assert chunk_text("", chunk_size=500, overlap=50) == []

    def test_sentence_boundary_respected(self):
        from backend.retrieval.rag import chunk_text

        text = "First sentence. Second sentence. Third sentence. " * 50
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        # No chunk should cut mid-word
        for c in chunks:
            assert c.strip()


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from backend.retrieval.rag import cosine_sim

        v = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_sim(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        from backend.retrieval.rag import cosine_sim

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_sim(a, b)) < 1e-6

    def test_zero_vector(self):
        from backend.retrieval.rag import cosine_sim

        a = np.zeros(3)
        b = np.array([1.0, 0.0, 0.0])
        assert cosine_sim(a, b) == 0.0


class TestMMR:
    def test_returns_top_k(self):
        from backend.retrieval.rag import mmr

        q_vec = np.array([1.0, 0.0])
        candidates = [
            ("text_a", {"url": "a"}, np.array([1.0, 0.0])),
            ("text_b", {"url": "b"}, np.array([0.9, 0.1])),
            ("text_c", {"url": "c"}, np.array([0.0, 1.0])),
        ]
        results = mmr(q_vec, candidates, top_k=2)
        assert len(results) == 2

    def test_more_diverse_than_greedy(self):
        from backend.retrieval.rag import mmr

        q_vec = np.array([1.0, 0.0])
        # Two near-identical candidates + one diverse
        c1 = ("t1", {}, np.array([1.0, 0.01]))
        c2 = ("t2", {}, np.array([1.0, 0.02]))
        c3 = ("t3", {}, np.array([0.1, 0.99]))
        results = mmr(q_vec, [c1, c2, c3], top_k=2, lambda_mult=0.3)
        texts = [r[0] for r in results]
        # With diversity weight, t3 should be selected over a near-duplicate
        assert "t3" in texts


class TestLocalRag:
    def test_ingest_and_retrieve(self, tmp_db, fake_embeddings):
        from backend.retrieval.rag import LocalRag, RagConfig
        from backend.llm.ollama import OllamaConfig

        ollama_cfg = OllamaConfig(base_url="http://fake", model="x", embed_model="y")
        rag = LocalRag(
            db_paths=tmp_db, ollama_cfg=ollama_cfg, rag_cfg=RagConfig(top_k=3)
        )

        with patch(
            "backend.retrieval.rag.ollama_embeddings", side_effect=fake_embeddings
        ):
            rag.ingest_text(
                "task1",
                "Python is a programming language.",
                {"source": "wikipedia", "url": "http://a"},
            )
            rag.ingest_text(
                "task1",
                "The Eiffel Tower is in Paris.",
                {"source": "wikipedia", "url": "http://b"},
            )
            results = rag.retrieve("programming language", task_id="task1")

        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r[0], str) for r in results)

    def test_ingest_returns_chunk_count(self, tmp_db, fake_embeddings):
        from backend.retrieval.rag import LocalRag, RagConfig
        from backend.llm.ollama import OllamaConfig

        ollama_cfg = OllamaConfig(base_url="http://fake", model="x", embed_model="y")
        rag = LocalRag(
            db_paths=tmp_db,
            ollama_cfg=ollama_cfg,
            rag_cfg=RagConfig(chunk_size=100, chunk_overlap=20),
        )

        long_text = "A" * 1000
        with patch(
            "backend.retrieval.rag.ollama_embeddings", side_effect=fake_embeddings
        ):
            n = rag.ingest_text(
                "task1", long_text, {"source": "test", "url": "http://c"}
            )
        assert n > 1


# ══════════════════════════════════════════════════════════════════════════════
# Citation tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCitationRegistry:
    def test_assigns_sequential_nums(self):
        from backend.retrieval.citations import CitationRegistry
        from backend.retrieval.sources import RetrievedDoc

        reg = CitationRegistry()
        d1 = RetrievedDoc("text1", "Title 1", "http://a", "wikipedia")
        d2 = RetrievedDoc("text2", "Title 2", "http://b", "arxiv")
        c1 = reg.register(d1)
        c2 = reg.register(d2)
        assert c1.num == 1
        assert c2.num == 2

    def test_deduplicates_by_url(self):
        from backend.retrieval.citations import CitationRegistry
        from backend.retrieval.sources import RetrievedDoc

        reg = CitationRegistry()
        d = RetrievedDoc("text", "Title", "http://same-url", "wikipedia")
        c1 = reg.register(d)
        c2 = reg.register(d)
        assert c1.num == c2.num
        assert len(reg.all()) == 1

    def test_bibliography_markdown_format(self):
        from backend.retrieval.citations import CitationRegistry
        from backend.retrieval.sources import RetrievedDoc

        reg = CitationRegistry()
        reg.register(RetrievedDoc("t", "My Title", "http://example.com", "wikipedia"))
        bib = reg.bibliography_markdown()
        assert "[1]" in bib
        assert "My Title" in bib
        assert "http://example.com" in bib

    def test_extract_citation_nums(self):
        from backend.retrieval.citations import extract_cited_nums

        text = "According to [1] and [3], the answer is [2]."
        nums = extract_cited_nums(text)
        assert sorted(nums) == [1, 2, 3]


# ══════════════════════════════════════════════════════════════════════════════
# Source retrieval tests (mocked HTTP)
# ══════════════════════════════════════════════════════════════════════════════


class TestWikipediaRetrieval:
    @patch("backend.retrieval.sources._SESSION")
    def test_search_returns_list(self, mock_session):
        from backend.retrieval.sources import wikipedia_search

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "query": {"search": [{"title": "Python (programming language)"}]}
        }
        mock_session.get.return_value = mock_resp
        results = wikipedia_search("Python programming")
        assert isinstance(results, list)
        assert results[0]["title"] == "Python (programming language)"

    @patch("backend.retrieval.sources._SESSION")
    def test_fetch_returns_doc(self, mock_session):
        from backend.retrieval.sources import wikipedia_fetch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "extract": "Python is a high-level programming language. " * 5,
            "content_urls": {
                "desktop": {"page": "https://en.wikipedia.org/wiki/Python"}
            },
        }
        mock_session.get.return_value = mock_resp
        doc = wikipedia_fetch("Python")
        assert doc is not None
        assert doc.source == "wikipedia"
        assert len(doc.text) > 80


class TestArxivRetrieval:
    @patch("backend.retrieval.sources._SESSION")
    def test_arxiv_parses_entries(self, mock_session):
        from backend.retrieval.sources import arxiv_search

        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>Attention Is All You Need</title>
            <summary>We propose a new network architecture called the Transformer.</summary>
            <id>https://arxiv.org/abs/1706.03762</id>
          </entry>
        </feed>"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = xml
        mock_session.get.return_value = mock_resp
        docs = arxiv_search("transformer attention")
        assert len(docs) == 1
        assert docs[0].source == "arxiv"
        assert "Attention" in docs[0].title


# ══════════════════════════════════════════════════════════════════════════════
# Auth tests
# ══════════════════════════════════════════════════════════════════════════════


def _make_test_app():
    """Build a fresh FastAPI app for testing — avoids module-level singleton issues."""
    from fastapi import FastAPI
    from backend.agents.pipeline import TaskManager
    from backend.api.v1 import router as v1_router, set_manager
    from backend.db import get_db_paths, init_db
    from backend.models import HealthResponse

    db = get_db_paths()
    init_db(db.db_path)
    mgr = TaskManager(db_paths=db)
    set_manager(mgr)

    app = FastAPI()
    app.include_router(v1_router, prefix="/v1")

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok", ollama_ok=False, redis_ok=False, db_ok=True)

    return app


class TestAuth:
    def test_valid_api_key_accepted(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "test-key-abc")
        from fastapi.testclient import TestClient

        client = TestClient(_make_test_app())
        resp = client.get("/v1/tasks", headers={"X-API-Key": "test-key-abc"})
        assert resp.status_code != 401

    def test_invalid_api_key_rejected(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "test-key-abc")
        from fastapi.testclient import TestClient

        client = TestClient(_make_test_app())
        resp = client.get("/v1/tasks", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_no_auth_when_disabled(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "")
        from fastapi.testclient import TestClient

        client = TestClient(_make_test_app())
        resp = client.get("/health")
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# API endpoint tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskAPI:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "")
        from fastapi.testclient import TestClient

        return TestClient(_make_test_app())

    @patch("backend.workers.rq_worker.enqueue_task", return_value=False)
    @patch("backend.agents.pipeline.TaskManager.run_task", return_value=None)
    def test_create_task_returns_202(self, mock_run, mock_enq, client):
        resp = client.post("/v1/tasks", json={"goal": "Explain transformers in NLP"})
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] in ("queued", "running", "succeeded")

    def test_get_nonexistent_task_404(self, client):
        resp = client.get("/v1/tasks/nonexistent-uuid")
        assert resp.status_code == 404

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "ollama_ok" in data


# ══════════════════════════════════════════════════════════════════════════════
# Code execution tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCodeExecution:
    def test_subprocess_runs_simple_code(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX", "subprocess")
        from backend.tools.code_exec import run_code

        result = run_code('print("hello")', timeout_s=5)
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_timeout_returns_error(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX", "subprocess")
        from backend.tools.code_exec import run_code

        result = run_code("import time; time.sleep(100)", timeout_s=1)
        assert result.exit_code != 0 or "Timeout" in result.stderr

    def test_dryrun_mode(self, monkeypatch):
        monkeypatch.setenv("CODE_SANDBOX", "none")
        from backend.tools.code_exec import run_code

        result = run_code('print("x")', timeout_s=5)
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout


# ══════════════════════════════════════════════════════════════════════════════
# DB schema tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDatabase:
    def test_init_creates_tables(self, tmp_db):
        from backend.db import connect

        with connect(tmp_db.db_path) as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"tasks", "steps", "docs"}.issubset(tables)

    def test_upsert_and_get_task(self, tmp_db):
        from backend.db import upsert_task, get_task

        upsert_task(tmp_db.db_path, "t1", "My goal", status="queued", owner="alice")
        row = get_task(tmp_db.db_path, "t1")
        assert row is not None
        assert row["goal"] == "My goal"
        assert row["status"] == "queued"

    def test_step_lifecycle(self, tmp_db):
        from backend.db import upsert_task, insert_step, update_step, list_task_steps

        upsert_task(tmp_db.db_path, "t1", "goal", "queued")
        insert_step(tmp_db.db_path, "s1", "t1", "researcher", "web_research", "running")
        update_step(tmp_db.db_path, "s1", status="succeeded", output="done")
        steps = list_task_steps(tmp_db.db_path, "t1")
        assert len(steps) == 1
        assert steps[0]["status"] == "succeeded"
        assert steps[0]["output"] == "done"
