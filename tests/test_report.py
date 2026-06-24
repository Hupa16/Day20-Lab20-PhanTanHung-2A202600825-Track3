from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.report import render_markdown_report


def test_report_renders_markdown() -> None:
    report = render_markdown_report(
        [
            BenchmarkMetrics(
                run_name="baseline",
                latency_seconds=1.23,
                input_tokens=10,
                output_tokens=5,
                citation_coverage=0.5,
            )
        ]
    )
    assert "Benchmark Report" in report
    assert "baseline" in report
    assert "Input tok" in report
    assert "50%" in report
