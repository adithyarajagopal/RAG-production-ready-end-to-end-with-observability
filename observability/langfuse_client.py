"""Langfuse client singleton — lazy-initialized on first get_client() call.

Reads credentials from environment variables:
  LANGFUSE_SECRET_KEY
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_BASE_URL

If any key is missing, get_client() returns None and the rest of the
observability layer degrades gracefully (no crashes, just no tracing).
"""

from __future__ import annotations

import os

_client = None
_initialized = False


def get_client():
    """Return the shared Langfuse client, or None if credentials are missing."""
    global _client, _initialized

    if _initialized:
        return _client

    _initialized = True

    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    base_url = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not secret_key or not public_key:
        _client = None
        return _client

    from langfuse import Langfuse

    _client = Langfuse(
        secret_key=secret_key,
        public_key=public_key,
        host=base_url,
    )
    return _client
