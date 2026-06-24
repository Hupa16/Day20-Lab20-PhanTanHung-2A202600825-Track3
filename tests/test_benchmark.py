from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    calculate_citation_coverage,
    run_benchmark,
    summarize_metrics,
)


def _finished_state() -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain GraphRAG for agents"))
    state.sources = [
        SourceDocument(title="Source 1", url="https://example.com/1", snippet="One"),
        SourceDocument(title="Source 2", url="https://example.com/2", snippet="Two"),
    ]
    state.research_notes = "Research"
    state.analysis_notes = "Analysis"
    state.final_answer = "GraphRAG helps agents reason over connected data [1]."
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=state.final_answer,
            metadata={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01},
        )
    )
    return state


def test_calculate_citation_coverage() -> None:
    assert calculate_citation_coverage(_finished_state()) == 0.5


def test_summarize_metrics_from_state() -> None:
    metrics = summarize_metrics("multi-agent", _finished_state(), latency_seconds=1.2)

    assert metrics.input_tokens == 10
    assert metrics.output_tokens == 5
    assert metrics.estimated_cost_usd == 0.01
    assert metrics.citation_coverage == 0.5
    assert metrics.failure_rate == 0.0


def test_run_benchmark_records_runner_failure() -> None:
    def failing_runner(query: str) -> ResearchState:
        raise RuntimeError(f"boom: {query}")

    state, metrics = run_benchmark("bad-run", "Explain GraphRAG", failing_runner)

    assert state.errors == ["boom: Explain GraphRAG"]
    assert metrics.failure_rate == 1.0
