"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings


def configure_tracing(settings: Settings | None = None) -> bool:
    """Configure LangSmith tracing from Settings."""

    settings = settings or get_settings()
    if os.environ.get("MALAB_DISABLE_REMOTE_TRACING", "").lower() == "true":
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    if not settings.langsmith_api_key:
        return False

    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    return True


@contextmanager
def tracing_session(
    name: str,
    metadata: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> Iterator[dict[str, Any]]:
    """Create a top-level tracing session with optional LangSmith context."""

    settings = settings or get_settings()
    enabled = configure_tracing(settings)
    attributes = {"session": name, **(metadata or {})}

    if not enabled:
        with trace_span(name, attributes) as span:
            yield span
        return

    try:
        from langsmith.run_helpers import tracing_context
    except ImportError:
        with trace_span(name, attributes) as span:
            span["langsmith_enabled"] = False
            yield span
        return

    with tracing_context(
        project_name=settings.langsmith_project,
        metadata=attributes,
        enabled=True,
    ), trace_span(name, attributes) as span:
        span["langsmith_enabled"] = True
        yield span


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    run_type: str = "chain",
) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton and LangSmith when enabled."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    if not _langsmith_enabled():
        try:
            yield span
        finally:
            span["duration_seconds"] = perf_counter() - started
        return

    try:
        from langsmith.run_helpers import trace
    except ImportError:
        try:
            yield span
        finally:
            span["duration_seconds"] = perf_counter() - started
        return

    manager = trace(name, run_type=run_type, inputs=attributes or {})
    try:
        run = manager.__enter__()
    except Exception:
        try:
            yield span
        finally:
            span["duration_seconds"] = perf_counter() - started
        return

    try:
        yield span
    except BaseException as exc:
        span["duration_seconds"] = perf_counter() - started
        run.outputs = {"duration_seconds": span["duration_seconds"]}
        suppress = manager.__exit__(type(exc), exc, exc.__traceback__)
        if not suppress:
            raise
    else:
        span["duration_seconds"] = perf_counter() - started
        run.outputs = {"duration_seconds": span["duration_seconds"]}
        manager.__exit__(None, None, None)


def _langsmith_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
