"""Multi-agent workflow orchestration."""

from collections.abc import Mapping

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span, tracing_session


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph shape.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self, agents: Mapping[str, BaseAgent] | None = None) -> None:
        self.agents = dict(agents or self._default_agents())

    def build(self) -> dict[str, object]:
        """Return a graph description used by the lightweight runner.

        This keeps the lab runnable without requiring LangGraph at first. The
        same nodes and conditional routes can be ported to LangGraph later.
        """

        return {
            "nodes": list(self.agents),
            "entrypoint": "supervisor",
            "conditional_routes": {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": None,
            },
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return final state."""

        self.build()
        with tracing_session(
            "multi_agent_workflow",
            {"query": state.request.query, "max_sources": state.request.max_sources},
        ) as workflow_span:
            state.add_trace_event(
                "workflow.started",
                {
                    "langsmith_enabled": workflow_span.get("langsmith_enabled", False),
                    "query": state.request.query,
                },
            )

            while True:
                with trace_span(
                    "supervisor",
                    {"iteration": state.iteration, "query": state.request.query},
                ) as span:
                    state = self.agents["supervisor"].run(state)
                state.trace[-1]["payload"]["duration_seconds"] = span["duration_seconds"]
                route = state.route_history[-1]

                if route == "done":
                    state.add_trace_event("workflow.completed", {"iteration": state.iteration})
                    return state

                agent = self.agents.get(route)
                if agent is None:
                    message = f"Supervisor selected unknown route: {route}"
                    state.errors.append(message)
                    state.add_trace_event("workflow.failed", {"error": message})
                    raise AgentExecutionError(message)

                try:
                    with trace_span(
                        route,
                        {"iteration": state.iteration, "route": route},
                    ) as span:
                        state = agent.run(state)
                    state.trace[-1]["payload"]["duration_seconds"] = span["duration_seconds"]
                except Exception as exc:
                    message = f"{route} failed: {exc}"
                    state.errors.append(message)
                    state.add_trace_event(
                        "workflow.failed",
                        {"route": route, "error": str(exc)},
                    )
                    return state

    def _default_agents(self) -> dict[str, BaseAgent]:
        return {
            "supervisor": SupervisorAgent(),
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }
