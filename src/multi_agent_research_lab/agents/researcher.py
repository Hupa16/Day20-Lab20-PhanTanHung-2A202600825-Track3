"""Researcher agent skeleton."""

from typing import Protocol

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class _SearchClient(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for source documents."""


class _LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return an LLM completion."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: _SearchClient | None = None,
        llm_client: _LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.
        """

        sources = self.search_client.search(
            state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources = sources

        source_block = "\n".join(
            f"[{index}] {source.title}\nURL: {source.url or 'N/A'}\nSnippet: {source.snippet}"
            for index, source in enumerate(sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are the Researcher in a multi-agent research workflow. "
                "Create concise research notes from supplied sources only. "
                "Keep source numbers visible for later citation."
            ),
            user_prompt=(
                f"Research question: {state.request.query}\n\n"
                f"Audience: {state.request.audience}\n\n"
                f"Sources:\n{source_block}\n\n"
                "Return bullet notes with source references like [1], [2]."
            ),
        )

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "source_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher.completed",
            {"source_count": len(sources), "output_tokens": response.output_tokens},
        )
        return state
