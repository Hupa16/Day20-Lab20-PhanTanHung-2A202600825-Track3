import pytest

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


class FakeLLMClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(
            content=self.content,
            input_tokens=len(system_prompt.split()) + len(user_prompt.split()),
            output_tokens=len(self.content.split()),
        )


class FakeSearchClient:
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [
            SourceDocument(
                title="GraphRAG overview",
                url="https://example.com/graphrag",
                snippet=f"{query} uses graph structure for retrieval.",
                metadata={"score": 0.9},
            )
        ][:max_results]


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain GraphRAG for agents"))


def test_researcher_populates_sources_and_notes() -> None:
    state = _state()
    agent = ResearcherAgent(
        search_client=FakeSearchClient(),
        llm_client=FakeLLMClient("Research notes [1]"),
    )

    result = agent.run(state)

    assert len(result.sources) == 1
    assert result.research_notes == "Research notes [1]"
    assert result.agent_results[-1].agent == AgentName.RESEARCHER
    assert result.trace[-1]["name"] == "researcher.completed"


def test_analyst_populates_analysis_notes() -> None:
    state = _state()
    state.research_notes = "Research notes [1]"
    agent = AnalystAgent(llm_client=FakeLLMClient("Analysis notes"))

    result = agent.run(state)

    assert result.analysis_notes == "Analysis notes"
    assert result.agent_results[-1].agent == AgentName.ANALYST
    assert result.trace[-1]["name"] == "analyst.completed"


def test_analyst_requires_research_notes() -> None:
    with pytest.raises(ValidationError):
        AnalystAgent(llm_client=FakeLLMClient("Analysis notes")).run(_state())


def test_writer_populates_final_answer() -> None:
    state = _state()
    state.sources = [
        SourceDocument(
            title="GraphRAG overview",
            url="https://example.com/graphrag",
            snippet="GraphRAG uses graph structure.",
        )
    ]
    state.research_notes = "Research notes [1]"
    state.analysis_notes = "Analysis notes"
    agent = WriterAgent(llm_client=FakeLLMClient("Final answer [1]"))

    result = agent.run(state)

    assert result.final_answer == "Final answer [1]"
    assert result.agent_results[-1].agent == AgentName.WRITER
    assert result.trace[-1]["name"] == "writer.completed"


def test_writer_requires_research_and_analysis_notes() -> None:
    with pytest.raises(ValidationError):
        WriterAgent(llm_client=FakeLLMClient("Final answer")).run(_state())
