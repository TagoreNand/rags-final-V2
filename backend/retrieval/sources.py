"""
backend/retrieval/sources.py

Multi-source web retrieval with:
  - Wikipedia (free, no key)
  - arXiv (free, no key)
  - Brave Search API (optional key)
  - SerpAPI / Google (optional key)
  - Generic URL fetch with BeautifulSoup extraction
  - Retry + timeout policies via tenacity
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from backend.config import get_settings
from backend.tracing import get_logger, timed

log = get_logger(__name__)


# ── Shared session with retry ─────────────────────────────────────────────────


def _make_session(max_retries: int = 3, backoff: float = 1.0) -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=max_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {"User-Agent": "agentic-rag-ops/2.0 (+https://github.com/yourorg/rag-ops)"}
    )
    return session


_SESSION = _make_session()


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class RetrievedDoc:
    """Normalised document returned by any source."""

    text: str
    title: str
    url: str
    source: str  # "wikipedia" | "arxiv" | "brave" | "url"
    score: float = 1.0  # relevance hint (0-1), if available
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            **self.extra,
        }


# ── Utility ───────────────────────────────────────────────────────────────────


def _clean(text: str, max_chars: Optional[int] = None) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if max_chars:
        t = t[:max_chars]
    return t


def _html_to_text(html: str, max_chars: int = 25_000) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(
        ["script", "style", "noscript", "nav", "footer", "header"]
    ):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    candidates = []
    for sel in [
        "article",
        "main",
        "[role=main]",
        "div[class*=content]",
        "div[class*=article]",
    ]:
        for node in soup.select(sel):
            t = node.get_text(" ", strip=True)
            if len(t) > 200:
                candidates.append(t)
    text = max(candidates, key=len) if candidates else soup.get_text(" ", strip=True)
    text = " ".join(text.split())
    if title and title not in text[:200]:
        text = f"{title}. {text}"
    return text[:max_chars]


# ── Wikipedia ─────────────────────────────────────────────────────────────────


@timed
def wikipedia_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    cfg = get_settings()
    resp = _SESSION.get(
        cfg.wiki_search_endpoint,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
            "srlimit": str(limit),
        },
        timeout=cfg.http_timeout_s,
    )
    resp.raise_for_status()
    hits = resp.json().get("query", {}).get("search", [])
    return [
        {
            "title": h["title"],
            "page_url": f"https://en.wikipedia.org/wiki/{h['title'].replace(' ', '_')}",
        }
        for h in hits
        if h.get("title")
    ]


@timed
def wikipedia_fetch(title: str) -> Optional[RetrievedDoc]:
    cfg = get_settings()
    url = f"{cfg.wiki_summary_endpoint}/{quote_plus(title)}"
    resp = _SESSION.get(url, timeout=cfg.http_timeout_s)
    if resp.status_code != 200:
        return None
    data = resp.json()
    extract = _clean(data.get("extract") or "", max_chars=cfg.max_web_fetch_chars)
    if len(extract) < 80:
        return None
    page_url = (
        data.get("content_urls", {})
        .get("desktop", {})
        .get("page", f"https://en.wikipedia.org/wiki/{title}")
    )
    return RetrievedDoc(text=extract, title=title, url=page_url, source="wikipedia")


# ── arXiv ────────────────────────────────────────────────────────────────────


@timed
def arxiv_search(query: str, max_results: int = 3) -> List[RetrievedDoc]:
    """Search arXiv Atom feed – no API key required."""
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    try:
        resp = _SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("arxiv_error", error=str(exc))
        return []

    docs = []
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        if title_el is None or summary_el is None:
            continue
        title = _clean(title_el.text or "")
        abstract = _clean(summary_el.text or "", max_chars=4000)
        arxiv_url = (id_el.text or "").strip()
        text = f"Title: {title}\n\nAbstract: {abstract}"
        docs.append(
            RetrievedDoc(
                text=text,
                title=title,
                url=arxiv_url,
                source="arxiv",
                extra={"type": "academic_paper"},
            )
        )
    return docs


# ── Brave Search ──────────────────────────────────────────────────────────────


@timed
def brave_search(query: str, count: int = 5) -> List[RetrievedDoc]:
    cfg = get_settings()
    if not cfg.brave_search_key:
        return []
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": cfg.brave_search_key,
    }
    try:
        resp = _SESSION.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params={"q": query, "count": count, "text_decorations": False},
            timeout=cfg.http_timeout_s,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.warning("brave_error", error=str(exc))
        return []

    docs = []
    for result in resp.json().get("web", {}).get("results", []):
        title = result.get("title", "")
        url = result.get("url", "")
        description = _clean(result.get("description", ""), max_chars=1000)
        if not url or not description:
            continue
        docs.append(
            RetrievedDoc(text=description, title=title, url=url, source="brave")
        )
    return docs


# ── Generic URL fetch ─────────────────────────────────────────────────────────


@timed
def fetch_url(url: str) -> Optional[RetrievedDoc]:
    cfg = get_settings()
    try:
        resp = _SESSION.get(url, timeout=cfg.http_timeout_s)
        resp.raise_for_status()
        text = _html_to_text(resp.text, max_chars=cfg.max_web_fetch_chars)
        soup = BeautifulSoup(resp.text, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else url
        return RetrievedDoc(text=text, title=title, url=url, source="url")
    except Exception as exc:
        log.warning("fetch_url_error", url=url, error=str(exc))
        return None


# ── Aggregator ────────────────────────────────────────────────────────────────


def multi_source_search(query: str, max_docs: int = 6) -> List[RetrievedDoc]:
    """
    Fan-out across all configured sources, deduplicate by URL, return up to max_docs.
    Priority: wikipedia > brave > arxiv (most factual first for general queries).
    """
    cfg = get_settings()
    docs: List[RetrievedDoc] = []
    seen_urls: set = set()

    def _add(d: Optional[RetrievedDoc]) -> None:
        if d and d.url not in seen_urls and len(d.text) > 80:
            seen_urls.add(d.url)
            docs.append(d)

    # Wikipedia
    for hit in wikipedia_search(query, limit=3):
        if len(docs) >= max_docs:
            break
        _add(wikipedia_fetch(hit["title"]))

    # Brave
    if len(docs) < max_docs:
        for bdoc in brave_search(query, count=3):
            if len(docs) >= max_docs:
                break
            _add(bdoc)

    # arXiv (when query seems academic)
    if cfg.arxiv_enabled and len(docs) < max_docs:
        for adoc in arxiv_search(query, max_results=2):
            if len(docs) >= max_docs:
                break
            _add(adoc)

    log.info(
        "retrieval_done",
        query=query,
        n_docs=len(docs),
        sources=list({d.source for d in docs}),
    )
    return docs[:max_docs]
