"""Command-line entrypoint for the lab starter."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_tracing
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""

    _init()
    state = _run_baseline(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


def _run_baseline(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    response = LLMClient().complete(
        system_prompt=(
            "You are a single-agent research assistant. Answer directly and clearly. "
            "If sources are not available, say what evidence would be needed."
        ),
        user_prompt=f"Research question: {query}\nAudience: {request.audience}",
    )
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    state.add_trace_event("baseline.completed", {"output_tokens": response.output_tokens})
    return state


def _run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=True))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Markdown report path"),
    ] = Path("reports/benchmark_report.md"),
) -> None:
    """Benchmark single-agent baseline against the multi-agent workflow."""

    _init()
    output.parent.mkdir(parents=True, exist_ok=True)
    _, baseline_metrics = run_benchmark("baseline", query, _run_baseline)
    _, multi_metrics = run_benchmark("multi-agent", query, _run_multi_agent)
    report = render_markdown_report([baseline_metrics, multi_metrics])
    output.write_text(report, encoding="utf-8")
    console.print(Panel.fit(str(output), title="Benchmark Report Written"))


if __name__ == "__main__":
    app()
