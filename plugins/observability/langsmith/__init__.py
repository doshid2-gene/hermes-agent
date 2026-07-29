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
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Trace tree: one root chain per turn, one llm child per LLM call, tools nested
# under their turn - the exact TaskTrace shape. Helpers take an injected client
# so they are unit-testable with a spy (no SDK, no network). Never raise.
# ---------------------------------------------------------------------------

@dataclass
class TraceState:
    root_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    current_turn_id: Optional[str] = None
    tools_by_id: Dict[str, str] = field(default_factory=dict)
    pending_tools_by_name: Dict[str, List[str]] = field(default_factory=dict)
    num_turns: int = 0
    used_tools: set = field(default_factory=set)
    is_error: bool = False
    error: Optional[str] = None
    answer: str = ""
    last_updated_at: float = field(default_factory=time.time)


def _usage_metadata(usage: Optional[dict]) -> Dict[str, Any]:
    """Map a CanonicalUsage summary dict to TaskTrace's usage_metadata shape."""
    if not usage:
        return {}
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    details = {
        k: v
        for k, v in (
            ("cache_read", usage.get("cache_read_tokens")),
            ("cache_creation", usage.get("cache_write_tokens")),
        )
        if v is not None
    }
    um: Dict[str, Any] = {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": (inp or 0) + (out or 0),
    }
    if details:
        um["input_token_details"] = details
    return {k: v for k, v in um.items() if v is not None}


def _start_root(client: Any, *, task_id: str, session_id: str, user_query: Any,
                metadata: Dict[str, Any]) -> TraceState:
    root_id = uuid.uuid4().hex
    clean = {k: v for k, v in metadata.items() if v is not None}
    try:
        client.create_run(
            id=root_id,
            name=f"task:{task_id or session_id or 'session'}",
            run_type="chain",
            inputs={"question": _jsonable(user_query)},
            project_name=_project(),
            start_time=datetime.now(UTC),
            extra={"metadata": clean},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith start_root failed: %s", exc)
    return TraceState(root_id=root_id, metadata=clean)


def _record_turn(client: Any, state: TraceState, *, index: int, model: Optional[str],
                 thinking: str, text: str, usage: Optional[dict],
                 started_at: datetime, ended_at: datetime) -> None:
    turn_id = uuid.uuid4().hex
    out_msg: Dict[str, Any] = {"role": "assistant", "content": text}
    if thinking:
        out_msg["thinking"] = _jsonable(thinking)
    extra: Dict[str, Any] = {"metadata": {"model": model, "turn": index}}
    um = _usage_metadata(usage)
    if um:
        extra["metadata"]["usage_metadata"] = um
    try:
        client.create_run(
            id=turn_id,
            name=f"turn {index}",
            run_type="llm",
            inputs={"messages": [{"role": "user", "content": "(agent turn)"}]},
            outputs={"messages": [out_msg]},
            parent_run_id=state.root_id,
            project_name=_project(),
            start_time=started_at,
            end_time=ended_at,
            extra=extra,
        )
        state.current_turn_id = turn_id
        state.num_turns = max(state.num_turns, index)
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith record_turn failed: %s", exc)


def _start_tool(client: Any, state: TraceState, *, tool_name: str, args: Any,
                tool_call_id: str, started_at: datetime) -> None:
    parent = state.current_turn_id or state.root_id
    tool_id = uuid.uuid4().hex
    try:
        client.create_run(
            id=tool_id,
            name=tool_name or "tool",
            run_type="tool",
            inputs={"input": _jsonable(args)},
            parent_run_id=parent,
            project_name=_project(),
            start_time=started_at,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith start_tool failed: %s", exc)
        return
    if tool_name:
        state.used_tools.add(tool_name)
    if tool_call_id:
        state.tools_by_id[tool_call_id] = tool_id
    else:
        state.pending_tools_by_name.setdefault(tool_name, []).append(tool_id)


def _end_tool(client: Any, state: TraceState, *, tool_name: str, result: Any,
              tool_call_id: str, is_error: bool, ended_at: datetime) -> None:
    tool_id = None
    if tool_call_id:
        tool_id = state.tools_by_id.pop(tool_call_id, None)
    if tool_id is None:
        queue = state.pending_tools_by_name.get(tool_name)
        if queue:
            tool_id = queue.pop(0)
            if not queue:
                state.pending_tools_by_name.pop(tool_name, None)
    if tool_id is None:
        return
    try:
        client.update_run(
            run_id=tool_id,
            outputs={"output": _jsonable(result)},
            end_time=ended_at,
            error="tool returned is_error" if is_error else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith end_tool failed: %s", exc)


def _finish(client: Any, state: TraceState, *, answer: str, num_turns: Optional[int],
            used_tools: Optional[list], is_error: bool, error: Optional[str],
            session_id: Optional[str], total_cost_usd: Optional[float]) -> None:
    md = {
        **state.metadata,
        "num_turns": num_turns if num_turns is not None else state.num_turns,
        "used_tools": used_tools if used_tools is not None else sorted(state.used_tools),
        "session_id": session_id,
        "is_error": is_error,
    }
    if total_cost_usd is not None:
        md["total_cost_usd"] = total_cost_usd
    try:
        client.update_run(
            run_id=state.root_id,
            outputs={"answer": _jsonable(answer)},
            end_time=datetime.now(UTC),
            error=error if is_error else None,
            extra={"metadata": md},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith finish failed: %s", exc)
