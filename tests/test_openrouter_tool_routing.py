"""OpenRouter fans a model slug out across multiple upstream providers, and not
all of them support tool calling. Without provider.require_parameters,
OpenRouter can route a tool-bound request to one that doesn't, which 404s with
"No endpoints found that support tool use." This pins that every OpenRouter
request opts into provider-level filtering on required parameters, and that
no other provider is affected.
"""

import pytest

from tradingagents.llm_clients.factory import create_llm_client


@pytest.mark.unit
def test_openrouter_requires_tool_capable_providers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = create_llm_client(
        provider="openrouter", model="anthropic/claude-sonnet-4.6"
    ).get_llm()
    assert llm.extra_body == {"provider": {"require_parameters": True}}


@pytest.mark.unit
@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-5.5"),
    ("groq", "llama-3.3-70b-versatile"),
    ("deepseek", "deepseek-chat"),
])
def test_other_providers_unaffected(monkeypatch, provider, model):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    llm = create_llm_client(provider=provider, model=model).get_llm()
    assert not getattr(llm, "extra_body", None)
