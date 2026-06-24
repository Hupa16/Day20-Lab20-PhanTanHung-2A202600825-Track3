"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and summarize observable quality signals."""

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:
        latency = perf_counter() - started
        state = ResearchState.model_validate({"request": {"query": query}, "errors": [str(exc)]})
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"Runner failed: {exc}",
        )
        return state, metrics

    latency = perf_counter() - started
    metrics = summarize_metrics(run_name, state, latency)
    return state, metrics


def summarize_metrics(
    run_name: str,
    state: ResearchState,
    latency_seconds: float,
) -> BenchmarkMetrics:
    """Create benchmark metrics from a finished state."""

    input_tokens = _sum_metadata_int(state, "input_tokens")
    output_tokens = _sum_metadata_int(state, "output_tokens")
    estimated_cost = _sum_metadata_float(state, "cost_usd")
    citation_coverage = calculate_citation_coverage(state)
    failure_rate = 1.0 if state.errors or not state.final_answer else 0.0
    quality_score = estimate_quality_score(state, citation_coverage)

    notes = _build_notes(state)
    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency_seconds,
        estimated_cost_usd=estimated_cost,
        quality_score=quality_score,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        citation_coverage=citation_coverage,
        failure_rate=failure_rate,
        notes=notes,
    )


def calculate_citation_coverage(state: ResearchState) -> float | None:
    """Estimate how many retrieved sources are cited in the final answer."""

    if not state.sources:
        return None
    if not state.final_answer:
        return 0.0

    cited_indexes = {
        int(match)
        for match in re.findall(r"\[(\d+)\]", state.final_answer)
        if int(match) >= 1
    }
    relevant_citations = {index for index in cited_indexes if index <= len(state.sources)}
    return len(relevant_citations) / len(state.sources)


def estimate_quality_score(
    state: ResearchState,
    citation_coverage: float | None,
) -> float | None:
    """Heuristic score for quick lab comparison before peer review."""

    if state.errors or not state.final_answer:
        return 0.0

    score = 4.0
    if state.research_notes:
        score += 1.5
    if state.analysis_notes:
        score += 1.5
    if len(state.final_answer.split()) >= 120:
        score += 1.0
    if citation_coverage is not None:
        score += min(citation_coverage, 1.0) * 2.0
    return min(score, 10.0)


def _sum_metadata_int(state: ResearchState, key: str) -> int | None:
    values = [
        value
        for result in state.agent_results
        if isinstance((value := result.metadata.get(key)), int)
    ]
    return sum(values) if values else None


def _sum_metadata_float(state: ResearchState, key: str) -> float | None:
    values = [
        value
        for result in state.agent_results
        if isinstance((value := result.metadata.get(key)), int | float)
    ]
    return float(sum(values)) if values else None


def _build_notes(state: ResearchState) -> str:
    if state.errors:
        return "; ".join(state.errors)

    parts = [
        f"routes={'>'.join(state.route_history)}",
        f"sources={len(state.sources)}",
    ]
    if state.final_answer:
        parts.append(f"answer_words={len(state.final_answer.split())}")
    return "; ".join(parts)
