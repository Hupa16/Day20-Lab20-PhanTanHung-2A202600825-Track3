"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routes are intentionally simple and state-driven:
        researcher -> analyst -> writer -> done.
        """

        if state.iteration >= self.settings.max_iterations:
            route = "done"
            reason = "max_iterations_reached"
        elif state.final_answer:
            route = "done"
            reason = "final_answer_ready"
        elif not state.research_notes:
            route = "researcher"
            reason = "missing_research_notes"
        elif not state.analysis_notes:
            route = "analyst"
            reason = "missing_analysis_notes"
        else:
            route = "writer"
            reason = "missing_final_answer"

        state.record_route(route)
        state.add_trace_event(
            "supervisor.route",
            {"next": route, "reason": reason, "iteration": state.iteration},
        )
        return state
