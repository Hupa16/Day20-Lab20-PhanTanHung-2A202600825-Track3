import json

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.services import search_client
from multi_agent_research_lab.services.search_client import SearchClient


class _FakeResponse:
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "results": [
                    {
                        "title": "GraphRAG paper",
                        "url": "https://example.com/graphrag",
                        "content": (
                            "GraphRAG combines graph structure with retrieval augmented generation."
                        ),
                        "score": 0.93,
                    }
                ],
                "response_time": "0.25",
                "usage": {"credits": 1},
                "request_id": "req-test",
            }
        ).encode("utf-8")


def test_search_client_maps_tavily_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse()

    monkeypatch.setattr(search_client, "urlopen", fake_urlopen)

    settings = Settings(TAVILY_API_KEY="tvly-test")
    documents = SearchClient(settings=settings).search("GraphRAG", max_results=1)

    assert documents[0].title == "GraphRAG paper"
    assert documents[0].url == "https://example.com/graphrag"
    assert documents[0].metadata["provider"] == "tavily"
    assert documents[0].metadata["usage"] == {"credits": 1}


def test_search_client_requires_tavily_key() -> None:
    settings = Settings(TAVILY_API_KEY="")

    with pytest.raises(ValidationError):
        SearchClient(settings=settings).search("GraphRAG", max_results=1)


def test_search_client_rejects_empty_query() -> None:
    settings = Settings(TAVILY_API_KEY="tvly-test")

    with pytest.raises(ValidationError):
        SearchClient(settings=settings).search("   ", max_results=1)
