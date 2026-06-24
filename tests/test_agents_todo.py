from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_first() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    result = SupervisorAgent().run(state)

    assert result.route_history == ["researcher"]
    assert result.iteration == 1
    assert result.trace[-1]["payload"]["reason"] == "missing_research_notes"


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Research notes"

    result = SupervisorAgent().run(state)

    assert result.route_history == ["analyst"]


def test_supervisor_routes_to_writer_after_analysis() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Research notes"
    state.analysis_notes = "Analysis notes"

    result = SupervisorAgent().run(state)

    assert result.route_history == ["writer"]


def test_supervisor_routes_to_done_when_final_answer_exists() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.final_answer = "Final answer"

    result = SupervisorAgent().run(state)

    assert result.route_history == ["done"]
    assert result.trace[-1]["payload"]["reason"] == "final_answer_ready"


def test_supervisor_stops_at_max_iterations() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.iteration = 1
    settings = Settings(MAX_ITERATIONS=1)

    result = SupervisorAgent(settings=settings).run(state)

    assert result.route_history == ["done"]
    assert result.trace[-1]["payload"]["reason"] == "max_iterations_reached"
