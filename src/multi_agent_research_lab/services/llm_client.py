"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.observability.tracing import trace_span


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Keep retry, timeout, and token logging here rather than inside agents.
        """

        if not self.settings.openai_api_key:
            raise ValidationError("OPENAI_API_KEY is required to call the LLM provider.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentExecutionError(
                'OpenAI SDK is not installed. Run: pip install -e ".[llm]"'
            ) from exc

        client = OpenAI(
            api_key=self.settings.openai_api_key,
            max_retries=2,
            timeout=float(self.settings.timeout_seconds),
        )

        try:
            with trace_span(
                "openai.chat.completions",
                {
                    "model": self.settings.openai_model,
                    "system_prompt_chars": len(system_prompt),
                    "user_prompt_chars": len(user_prompt),
                },
                run_type="llm",
            ):
                completion = client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
        except Exception as exc:
            raise AgentExecutionError(f"LLM completion failed: {exc}") from exc

        message = completion.choices[0].message
        content = message.content or ""
        usage = completion.usage

        return LLMResponse(
            content=content,
            input_tokens=None if usage is None else usage.prompt_tokens,
            output_tokens=None if usage is None else usage.completion_tokens,
            cost_usd=None,
        )
