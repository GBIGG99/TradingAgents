"""Tests for the Groq/OpenAI-compatible tool-call retry helper.

All chain invocation is mocked, so these run without a network connection
or a live LLM.
"""
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.utils.tool_call_retry import invoke_with_tool_retry


class _FakeAPIError(Exception):
    """Mimics openai.APIStatusError's relevant attributes without the SDK."""

    def __init__(self, code: str, status_code: int):
        super().__init__(f"fake api error: {code}")
        self.code = code
        self.status_code = status_code


@pytest.mark.unit
class TestToolUseFailedRetry:
    def test_succeeds_on_first_try_without_retry(self):
        chain = MagicMock()
        chain.invoke.return_value = "OK"
        result = invoke_with_tool_retry(chain, ["msg"], "Market Analyst")
        assert result == "OK"
        assert chain.invoke.call_count == 1

    def test_retries_on_malformed_tool_call_then_succeeds(self):
        chain = MagicMock()
        chain.invoke.side_effect = [
            _FakeAPIError("tool_use_failed", 400),
            "OK",
        ]
        result = invoke_with_tool_retry(chain, ["msg"], "Market Analyst")
        assert result == "OK"
        assert chain.invoke.call_count == 2

    def test_gives_up_after_max_retries(self):
        chain = MagicMock()
        chain.invoke.side_effect = _FakeAPIError("tool_use_failed", 400)
        with pytest.raises(_FakeAPIError):
            invoke_with_tool_retry(chain, ["msg"], "Market Analyst")
        # 1 initial attempt + 2 retries = 3 calls
        assert chain.invoke.call_count == 3


@pytest.mark.unit
class TestRateLimitHandling:
    def test_retries_once_on_429_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(
            "tradingagents.agents.utils.tool_call_retry.time.sleep", lambda s: None
        )
        chain = MagicMock()
        chain.invoke.side_effect = [
            _FakeAPIError("rate_limit_exceeded", 429),
            "OK",
        ]
        result = invoke_with_tool_retry(chain, ["msg"], "News Analyst")
        assert result == "OK"
        assert chain.invoke.call_count == 2

    def test_413_request_too_large_is_not_retried(self):
        # A 413 means the single request already exceeds the model's
        # per-minute budget; retrying the identical request can never
        # succeed, so this must raise immediately (call_count == 1) with an
        # actionable message rather than looping.
        chain = MagicMock()
        chain.invoke.side_effect = _FakeAPIError("rate_limit_exceeded", 413)
        with pytest.raises(RuntimeError, match="larger than the model's"):
            invoke_with_tool_retry(chain, ["msg"], "Fundamentals Analyst")
        assert chain.invoke.call_count == 1


@pytest.mark.unit
class TestUnrelatedErrorsPassThrough:
    def test_unrelated_exception_is_not_retried(self):
        chain = MagicMock()
        chain.invoke.side_effect = ValueError("something unrelated broke")
        with pytest.raises(ValueError, match="something unrelated broke"):
            invoke_with_tool_retry(chain, ["msg"], "Market Analyst")
        assert chain.invoke.call_count == 1

    def test_unknown_api_error_code_is_not_retried(self):
        chain = MagicMock()
        chain.invoke.side_effect = _FakeAPIError("some_other_error", 400)
        with pytest.raises(_FakeAPIError):
            invoke_with_tool_retry(chain, ["msg"], "Market Analyst")
        assert chain.invoke.call_count == 1
