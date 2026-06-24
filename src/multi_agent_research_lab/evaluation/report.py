"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Input tok | Output tok | "
        "Quality | Citations | Failures | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        input_tokens = "" if item.input_tokens is None else str(item.input_tokens)
        output_tokens = "" if item.output_tokens is None else str(item.output_tokens)
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citations = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | "
            f"{cost} | {input_tokens} | {output_tokens} | {quality} | "
            f"{citations} | {item.failure_rate:.0%} | {item.notes} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Quality is a lightweight heuristic for lab comparison, "
            "not a substitute for peer review.",
            "- Citation coverage counts unique source markers like `[1]` in the final answer.",
            "- Failure rate is `100%` when a run records errors or produces no final answer.",
        ]
    )
    return "\n".join(lines) + "\n"
