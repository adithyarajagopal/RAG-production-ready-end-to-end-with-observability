"""Helper functions wrapping Langfuse trace/span/generation/score APIs.

Every function is a no-op if the Langfuse client is None (graceful degradation).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from observability.langfuse_client import get_client


def create_trace(
    name: str,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    input: Any | None = None,
):
    """Start a new Langfuse trace for a request."""
    client = get_client()
    if client is None:
        return None

    return client.trace(
        name=name,
        user_id=user_id,
        metadata=metadata or {},
        input=input,
    )


def create_span(
    trace,
    name: str,
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Create a span within an existing trace."""
    if trace is None:
        return None

    return trace.span(
        name=name,
        input=input,
        metadata=metadata or {},
        start_time=datetime.now(timezone.utc),
    )


def end_span(
    span,
    output: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    """End a span with output data."""
    if span is None:
        return

    span.end(
        output=output,
        metadata=metadata or {},
        end_time=datetime.now(timezone.utc),
    )


def score_trace(
    trace,
    name: str,
    value: float,
    comment: str | None = None,
):
    """Attach a numeric score to a trace."""
    if trace is None:
        return

    trace.score(
        name=name,
        value=value,
        comment=comment,
    )


def create_generation(
    trace,
    name: str,
    model: str,
    input: Any | None = None,
    output: Any | None = None,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Log an LLM generation with token counts and cost.

    Args:
        trace: Parent trace (or None to skip).
        name: Generation name (e.g. "rag_generate").
        model: Model identifier (e.g. "anthropic/claude-sonnet-4").
        input: Messages sent to the LLM.
        output: LLM response text.
        usage: Dict with keys: prompt_tokens, completion_tokens, total_tokens.
        metadata: Additional metadata.
    """
    if trace is None:
        return None

    return trace.generation(
        name=name,
        model=model,
        input=input,
        output=output,
        usage=usage or {},
        metadata=metadata or {},
    )
