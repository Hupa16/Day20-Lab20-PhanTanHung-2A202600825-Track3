from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


class FakeSupervisor(BaseAgent):
    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        elif not state.final_answer:
            route = "writer"
        else:
            route = "done"
        state.record_route(route)
        return state


class FakeResearcher(BaseAgent):
    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        state.research_notes = "research"
        return state


class FakeAnalyst(BaseAgent):
    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        state.analysis_notes = "analysis"
        return state


class FakeWriter(BaseAgent):
    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        state.final_answer = "final"
        return state


class FailingResearcher(BaseAgent):
    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        raise RuntimeError("search unavailable")


class UnknownRouteSupervisor(BaseAgent):
    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        state.record_route("critic")
        return state


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_workflow_runs_agents_until_done() -> None:
    workflow = MultiAgentWorkflow(
        agents={
            "supervisor": FakeSupervisor(),
            "researcher": FakeResearcher(),
            "analyst": FakeAnalyst(),
            "writer": FakeWriter(),
        }
    )

    result = workflow.run(_state())

    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert result.final_answer == "final"
    assert result.trace[-1]["name"] == "workflow.completed"


def test_workflow_records_worker_failures() -> None:
    workflow = MultiAgentWorkflow(
        agents={
            "supervisor": FakeSupervisor(),
            "researcher": FailingResearcher(),
            "analyst": FakeAnalyst(),
            "writer": FakeWriter(),
        }
    )

    result = workflow.run(_state())

    assert result.errors == ["researcher failed: search unavailable"]
    assert result.trace[-1]["name"] == "workflow.failed"


def test_workflow_rejects_unknown_routes() -> None:
    workflow = MultiAgentWorkflow(agents={"supervisor": UnknownRouteSupervisor()})

    try:
        workflow.run(_state())
    except AgentExecutionError as exc:
        assert "unknown route" in str(exc)
    else:
        raise AssertionError("Expected AgentExecutionError")
