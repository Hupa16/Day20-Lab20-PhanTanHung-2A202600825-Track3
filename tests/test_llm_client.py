import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.services.llm_client import LLMClient


def test_llm_client_requires_openai_key() -> None:
    settings = Settings(OPENAI_API_KEY="")

    with pytest.raises(ValidationError):
        LLMClient(settings=settings).complete("system", "user")
