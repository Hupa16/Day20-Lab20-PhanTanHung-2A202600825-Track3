"""Search client abstraction for ResearcherAgent."""

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.observability.tracing import trace_span


class SearchClient:
    """Provider-agnostic search client skeleton."""

    endpoint = "https://api.tavily.com/search"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Uses Tavily Search and normalizes the response into SourceDocument objects.
        """

        if not self.settings.tavily_api_key:
            raise ValidationError("TAVILY_API_KEY is required to call the search provider.")
        if not query.strip():
            raise ValidationError("Search query cannot be empty.")

        bounded_max_results = max(1, min(max_results, 20))
        payload = {
            "query": query,
            "max_results": bounded_max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_usage": True,
        }

        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.tavily_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with trace_span(
                "tavily.search",
                {"query": query, "max_results": bounded_max_results},
                run_type="tool",
            ), urlopen(request, timeout=self.settings.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = f"Tavily search failed with HTTP {exc.code}: {detail}"
            raise AgentExecutionError(message) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AgentExecutionError(f"Tavily search failed: {exc}") from exc

        results = response_payload.get("results", [])
        if not isinstance(results, list):
            raise AgentExecutionError("Tavily search returned an invalid results payload.")

        return [
            self._to_source_document(item, response_payload)
            for item in results[:bounded_max_results]
        ]

    def _to_source_document(
        self, item: dict[str, Any], response_payload: dict[str, Any]
    ) -> SourceDocument:
        title = str(item.get("title") or "Untitled source")
        snippet = str(item.get("content") or "")
        url = item.get("url")

        return SourceDocument(
            title=title,
            url=url if isinstance(url, str) else None,
            snippet=snippet,
            metadata={
                "provider": "tavily",
                "score": item.get("score"),
                "raw_content": item.get("raw_content"),
                "favicon": item.get("favicon"),
                "response_time": response_payload.get("response_time"),
                "request_id": response_payload.get("request_id"),
                "usage": response_payload.get("usage"),
            },
        )
