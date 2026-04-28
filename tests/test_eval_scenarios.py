"""Deterministic scenario validation for evaluation harness thresholds."""

from __future__ import annotations

import json
from pathlib import Path
import re

from agents.graph import run_research, resolve_execution_budget


def _parse_reference_urls(summary: str) -> set[str]:
    """Extract citation URLs from generated summary text."""

    return set(re.findall(r"https?://[^\s]+", summary))


def test_deterministic_eval_scenarios_pass_thresholds(monkeypatch):
    """Bundled deterministic scenarios should satisfy configured verdict and score thresholds."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    scenario_path = Path("eval/scenarios/deterministic_eval.json")
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))

    for item in payload:
        execution_tier, source_budget = resolve_execution_budget(
            max_sources=int(item["max_sources"]),
            execution_tier=str(item["execution_tier"]),
        )
        result = run_research(
            query=str(item["query"]),
            max_sources=source_budget,
            execution_tier=execution_tier,
        )

        source_urls = [src["url"] for src in result["sources"] if src.get("url")]
        cited_urls = _parse_reference_urls(str(result["summary"]))
        citation_ratio = (sum(1 for url in source_urls if url in cited_urls) / len(source_urls)) if source_urls else 0.0

        assert result["critique"]["verdict"] == item["expected_verdict"]
        assert float(result["critique"]["confidence"]) >= float(item["expected_min_confidence"])
        assert citation_ratio >= float(item["expected_min_citation_match"])
