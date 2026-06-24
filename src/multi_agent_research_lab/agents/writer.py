"""Writer agent skeleton."""

from typing import Protocol

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


class _LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return an LLM completion."""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: _LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.
        """

        if not state.research_notes:
            raise ValidationError("WriterAgent requires research_notes before writing.")
        if not state.analysis_notes:
            raise ValidationError("WriterAgent requires analysis_notes before writing.")

        source_list = "\n".join(
            f"[{index}] {source.title} - {source.url or 'N/A'}"
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are the Writer in a multi-agent research workflow. "
                "Write a clear final answer for the target audience. "
                "Use citation markers that match the provided source list."
            ),
            user_prompt=(
                f"Research question: {state.request.query}\n\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                f"Analysis notes:\n{state.analysis_notes}\n\n"
                f"Sources:\n{source_list}\n\n"
                "Write the final answer with concise paragraphs and citations."
            ),
        )

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "source_count": len(state.sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "writer.completed",
            {"source_count": len(state.sources), "output_tokens": response.output_tokens},
        )
        return state
