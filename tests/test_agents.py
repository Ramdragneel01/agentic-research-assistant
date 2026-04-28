
"""Unit tests for Searcher, Summarizer, and Critic agents."""

from __future__ import annotations

from agents.critic import critique
from agents.graph import run_research, resolve_execution_budget
from agents.searcher import search_sources
from agents.summarizer import summarize


def test_search_sources_returns_fallback_when_no_key(monkeypatch):
    """Searcher should return deterministic fallback sources without Tavily key."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    sources = search_sources("enterprise secrets management", limit=3)

    assert len(sources) == 3
    assert all("title" in item for item in sources)
    assert all("content" in item for item in sources)


def test_summarizer_includes_findings_and_references():
    """Summarizer should include expected structural sections for grounding."""

    summary = summarize(
        "evaluate migration strategy",
        [
            {
                "title": "NIST reference",
                "url": "https://example.com/nist",
                "content": "Detailed guidance on identity and controls.",
            },
            {
                "title": "OWASP reference",
                "url": "https://example.com/owasp",
                "content": "API and validation recommendations.",
            },
        ],
    )

    assert "Key Findings:" in summary
    assert "References:" in summary


def test_critic_detects_low_evidence_summary():
    """Critic should return revision verdict for low-information summaries."""

    payload = critique("No sources were retrieved for this query.", source_count=0)

    assert payload["verdict"] == "needs_revision"
    assert payload["confidence"] < 0.88
    assert len(payload["gaps"]) > 0


def test_execution_budget_resolution_by_tier():
    """Tier resolution should apply expected source budget caps."""

    assert resolve_execution_budget(max_sources=8, execution_tier="small") == ("small", 3)
    assert resolve_execution_budget(max_sources=8, execution_tier="medium") == ("medium", 5)
    assert resolve_execution_budget(max_sources=8, execution_tier="large") == ("large", 8)


def test_run_research_is_repeatable_for_same_query_without_live_search(monkeypatch):
    """Offline execution should produce stable outputs for the same query and settings."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    first = run_research(
        query="Evaluate zero trust governance priorities",
        max_sources=8,
        execution_tier="small",
    )
    second = run_research(
        query="Evaluate zero trust governance priorities",
        max_sources=8,
        execution_tier="small",
    )

    assert first["execution_tier"] == "small"
    assert first["source_budget"] == 3
    assert len(first["sources"]) <= 3
    assert first["summary"] == second["summary"]
    assert first["critique"]["verdict"] == second["critique"]["verdict"]
    assert [item["url"] for item in first["sources"]] == [item["url"] for item in second["sources"]]
