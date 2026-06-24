import os

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.observability.tracing import configure_tracing, trace_span


def test_configure_tracing_sets_langsmith_env(monkeypatch) -> None:
    monkeypatch.delenv("MALAB_DISABLE_REMOTE_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    settings = Settings(LANGSMITH_API_KEY="lsv2-test", LANGSMITH_PROJECT="test-project")

    enabled = configure_tracing(settings)

    assert enabled is True
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2-test"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_trace_span_records_duration_without_langsmith(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    with trace_span("unit-test") as span:
        span["ok"] = True

    assert span["ok"] is True
    assert isinstance(span["duration_seconds"], float)
