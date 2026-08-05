"""Retry helper for tool-calling invocations against providers with unreliable
JSON tool-call generation — notably Groq's Llama models, which sometimes emit
a malformed pseudo-XML tool call (``<function=name>args</function>``) instead
of a valid JSON ``tool_calls`` payload.

Groq/OpenAI-compatible errors surface here as ``openai.APIStatusError`` (or a
subclass), with ``.code`` and ``.status_code`` set from the response body.
Two distinct failure modes need different handling:

- ``tool_use_failed`` (HTTP 400): the model emitted a malformed tool call.
  This is usually a one-off sampling glitch, not a deterministic failure, so
  retrying the identical call is often enough to get a well-formed one.
- ``rate_limit_exceeded`` where the request itself exceeds the model's
  per-minute token budget (HTTP 413, "Request too large"): retrying the
  identical request can never succeed — it's bigger than the limit on its
  own, not just competing with recent usage — so this is raised immediately
  with an actionable message instead of being retried.
- A genuine rate limit (HTTP 429, near-but-not-over the cap): retried once
  after a short backoff, since waiting can actually help here.

Anything else is re-raised immediately; this module only handles known,
specifically-diagnosed transient/non-transient Groq tool-calling failures.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_USE_FAILED_RETRIES = 2
_RATE_LIMIT_RETRIES = 1
_RATE_LIMIT_BACKOFF_SECONDS = 5.0


def invoke_with_tool_retry(chain: Any, messages: Any, agent_name: str) -> Any:
    """Invoke a tool-bound chain, retrying known-transient tool-call failures."""
    attempt = 0
    while True:
        try:
            return chain.invoke(messages)
        except Exception as exc:
            code = getattr(exc, "code", None)
            status_code = getattr(exc, "status_code", None)

            if code == "tool_use_failed" and attempt < _TOOL_USE_FAILED_RETRIES:
                attempt += 1
                logger.warning(
                    "%s: model emitted a malformed tool call (attempt %d/%d); retrying",
                    agent_name, attempt, _TOOL_USE_FAILED_RETRIES,
                )
                continue

            if (
                code == "rate_limit_exceeded"
                and status_code == 429
                and attempt < _RATE_LIMIT_RETRIES
            ):
                attempt += 1
                logger.warning(
                    "%s: rate-limited (attempt %d/%d); waiting %.0fs before retrying",
                    agent_name, attempt, _RATE_LIMIT_RETRIES, _RATE_LIMIT_BACKOFF_SECONDS,
                )
                time.sleep(_RATE_LIMIT_BACKOFF_SECONDS)
                continue

            if code == "rate_limit_exceeded" and status_code == 413:
                raise RuntimeError(
                    f"{agent_name}: this request is larger than the model's "
                    "per-minute token budget, so retrying the same request "
                    "won't help. Try a model with a larger rate limit, or "
                    "fewer debate rounds."
                ) from exc

            raise
