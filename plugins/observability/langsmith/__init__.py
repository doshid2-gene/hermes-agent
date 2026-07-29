"""langsmith - Hermes plugin for LangSmith observability.

Traces Hermes conversations, LLM calls, and tool usage to LangSmith as the same
run-tree the deep-claude-agent baseline emits (``TaskTrace``):

    task:<id>              root run  run_type="chain"
     |- turn 1             child     run_type="llm"   (thinking + text + usage)
     |   |- <tool>         child     run_type="tool"  (input=args, output=result)
     |- turn 2             child     run_type="llm"
     ...

Activation is via the Hermes plugin system (``hermes plugins enable
observability/langsmith``). At runtime it also requires the ``langsmith`` SDK and an
API key; without either, every hook is a silent no-op (fail open).

Env (standard LangSmith names; matches the deep-claude-agent baseline so runs can
share a project):
  LANGSMITH_API_KEY / LANGCHAIN_API_KEY    -- enables tracing when present
  LANGSMITH_ENDPOINT / LANGCHAIN_ENDPOINT  -- self-hosted URL, used verbatim as api_url
  LANGSMITH_PROJECT / LANGCHAIN_PROJECT    -- project name (default: hermes-agent)

Roche self-hosted note: the endpoint expects the ``/api/v1`` suffix and a corporate
SSL workaround (verify=False), matching self-evo's ls_client().
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PROJECT = "hermes-agent"
_MAX_FIELD_CHARS = 40_000
_INIT_FAILED = object()
_LANGSMITH_CLIENT: Any = None


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _api_key() -> Optional[str]:
    return _env("LANGSMITH_API_KEY") or _env("LANGCHAIN_API_KEY") or None


def tracing_enabled() -> bool:
    return _api_key() is not None


def _project() -> str:
    return _env("LANGSMITH_PROJECT") or _env("LANGCHAIN_PROJECT") or _DEFAULT_PROJECT


def _jsonable(value: Any) -> str:
    """Best-effort JSON-safe, size-capped rendering of an input/output field."""
    try:
        text = value if isinstance(value, str) else json.dumps(
            value, default=str, ensure_ascii=False
        )
    except Exception:  # noqa: BLE001
        text = str(value)
    if len(text) > _MAX_FIELD_CHARS:
        text = text[:_MAX_FIELD_CHARS] + f"\n...[truncated {len(text) - _MAX_FIELD_CHARS} chars]"
    return text


def _client() -> Any:
    """Return a cached LangSmith Client, or None if unavailable/unconfigured.

    Matches self-evo ls_client(): disable insecure-request warnings, pass the
    endpoint verbatim as api_url (keep the /api/v1 suffix), and disable SSL verify
    for the Roche self-hosted server. Fail-open + cached.
    """
    global _LANGSMITH_CLIENT
    if _LANGSMITH_CLIENT is _INIT_FAILED:
        return None
    if _LANGSMITH_CLIENT is not None:
        return _LANGSMITH_CLIENT
    if not tracing_enabled():
        _LANGSMITH_CLIENT = _INIT_FAILED
        return None
    try:
        import urllib3
        from langsmith import Client
    except Exception:  # noqa: BLE001 - SDK not installed
        _LANGSMITH_CLIENT = _INIT_FAILED
        return None
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        endpoint = _env("LANGSMITH_ENDPOINT") or _env("LANGCHAIN_ENDPOINT")
        kwargs: Dict[str, Any] = {"api_key": _api_key()}
        if endpoint:
            kwargs["api_url"] = endpoint  # verbatim - keep /api/v1
        client = Client(**kwargs)
        try:
            client.session.verify = False
        except Exception:  # noqa: BLE001
            pass
        _LANGSMITH_CLIENT = client
        return client
    except Exception as exc:  # noqa: BLE001 - never break the run on client init
        logger.debug("LangSmith client init failed: %s", exc)
        _LANGSMITH_CLIENT = _INIT_FAILED
        return None
