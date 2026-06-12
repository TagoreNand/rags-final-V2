"""
backend/retrieval/citations.py

Rich citation management:
  - Assigns numeric citation IDs to every retrieved document
  - Formats a bibliography block (Markdown or JSON)
  - Injects citation-number context into retrieval blocks sent to the LLM
  - Parses [1], [2] markers from LLM output for downstream attribution
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from backend.retrieval.sources import RetrievedDoc


@dataclass
class Citation:
    num: int  # [1], [2], ...
    title: str
    url: str
    source: str  # "wikipedia" | "arxiv" | "brave" | "url"
    snippet: str  # first 300 chars of the text
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num": self.num,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "snippet": self.snippet,
            **self.extra,
        }

    def markdown_ref(self) -> str:
        return f"[{self.num}] **{self.title}** ({self.source})  \n{self.url}"


class CitationRegistry:
    """Tracks all sources used across the current task run."""

    def __init__(self) -> None:
        self._by_url: Dict[str, Citation] = {}
        self._counter: int = 0

    def register(self, doc: RetrievedDoc) -> Citation:
        if doc.url in self._by_url:
            return self._by_url[doc.url]
        self._counter += 1
        c = Citation(
            num=self._counter,
            title=doc.title,
            url=doc.url,
            source=doc.source,
            snippet=doc.text[:300],
            extra=doc.extra,
        )
        self._by_url[doc.url] = c
        return c

    def all(self) -> List[Citation]:
        return sorted(self._by_url.values(), key=lambda c: c.num)

    def bibliography_markdown(self) -> str:
        lines = ["## References\n"]
        for c in self.all():
            lines.append(c.markdown_ref())
        return "\n\n".join(lines)

    def bibliography_json(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.all()]


def format_context_with_citations(
    blocks: List[Tuple[str, Dict[str, Any]]],
    registry: CitationRegistry,
) -> str:
    """
    Formats retrieval blocks for the LLM prompt with embedded [N] citation
    numbers so the model can reference sources inline.
    """
    out: List[str] = []
    for text, meta in blocks:
        url = meta.get("url", "")
        title = meta.get("title", "Unknown")
        source = meta.get("source", "unknown")
        doc = RetrievedDoc(text=text, title=title, url=url, source=source, extra={})
        cite = registry.register(doc)
        out.append(f"[{cite.num}] Source: {title} ({source})\nURL: {url}\n\n{text}")
    return "\n\n---\n\n".join(out)


def extract_cited_nums(text: str) -> List[int]:
    """Returns list of citation numbers [N] found in LLM output."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", text)]
