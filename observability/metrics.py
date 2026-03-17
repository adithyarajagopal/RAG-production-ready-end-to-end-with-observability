"""Helper functions wrapping Langfuse v4 trace/span/generation/score APIs.

Every function is a no-op if the Langfuse client is None (graceful degradation).

Langfuse v4 API:
- client.start_observation(name, as_type="span"|"generation") → root observation
- parent.start_observation(name, as_type="span"|"generation") → nested child
- span.update(output=...) then span.end() → close a span
- span.score_trace(name, value, comment) → score the whole trace
"""

from __future__ import annotations

from typing import Any

from observability.langfuse_client import get_client


def create_trace(
    name: str,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    input: Any | None = None,
):
    """Start a new Langfuse trace (root observation) for a request."""
    client = get_client()
    if client is None:
        return None

    return client.start_observation(
        name=name,
        as_type="span",
        input=input,
        metadata=metadata or {},
    )


def create_span(
    trace,
    name: str,
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Create a child span within an existing trace/span."""
    if trace is None:
        return None

    return trace.start_observation(
        name=name,
        as_type="span",
        input=input,
        metadata=metadata or {},
    )


def end_span(
    span,
    output: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    """End a span with output data."""
    if span is None:
        return

    span.update(output=output, metadata=metadata or {})
    span.end()


def score_trace(
    trace,
    name: str,
    value: float,
    comment: str | None = None,
):
    """Attach a numeric score to the trace."""
    if trace is None:
        return

    trace.score_trace(
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
    """Log an LLM generation with token counts.

    Args:
        trace: Parent trace/span (or None to skip).
        name: Generation name (e.g. "rag_generate").
        model: Model identifier (e.g. "anthropic/claude-sonnet-4").
        input: Messages sent to the LLM.
        output: LLM response text.
        usage: Dict with keys: input, output, total (token counts).
        metadata: Additional metadata.
    """
    if trace is None:
        return None

    gen = trace.start_observation(
        name=name,
        as_type="generation",
        model=model,
        input=input,
        output=output,
        usage_details=usage or {},
        metadata=metadata or {},
    )
    gen.end()
    return gen
