"""Analyst agent skeleton."""

from typing import Protocol

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


class _LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return an LLM completion."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: _LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.
        """

        if not state.research_notes:
            raise ValidationError("AnalystAgent requires research_notes before analysis.")

        response = self.llm_client.complete(
            system_prompt=(
                "You are the Analyst in a multi-agent research workflow. "
                "Extract key claims, compare viewpoints, identify uncertainty, "
                "and flag weak evidence. Do not write the final answer."
            ),
            user_prompt=(
                f"Research question: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Return structured analysis with sections: Key claims, "
                "Evidence strength, Tensions or gaps, Recommended answer angle."
            ),
        )

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst.completed",
            {"output_tokens": response.output_tokens},
        )
        return state
