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
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PROJECT = "hermes-agent"
_MAX_FIELD_CHARS = 40_000
_INIT_FAILED = object()
_LANGSMITH_CLIENT: Any = None


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _api_key() -> str | None:
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
        kwargs: dict[str, Any] = {"api_key": _api_key()}
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
        logger.debug("LangSmith client init failed: %s", type(exc).__name__)
        _LANGSMITH_CLIENT = _INIT_FAILED
        return None


# ---------------------------------------------------------------------------
# Trace tree: one root chain per turn, one llm child per LLM call, tools nested
# under their turn - the exact TaskTrace shape. Helpers take an injected client
# so they are unit-testable with a spy (no SDK, no network). Never raise.
# ---------------------------------------------------------------------------

@dataclass
class TraceState:
    # ``root_id is None`` marks a no-op trace (root create_run failed): every
    # helper short-circuits so we never emit children under a nonexistent parent.
    root_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    current_turn_id: str | None = None
    tools_by_id: dict[str, str] = field(default_factory=dict)
    pending_tools_by_name: dict[str, list[str]] = field(default_factory=dict)
    num_turns: int = 0
    used_tools: set[str] = field(default_factory=set)
    is_error: bool = False
    error: str | None = None
    answer: str = ""
    last_updated_at: float = field(default_factory=time.time)


def _usage_metadata(usage: dict | None) -> dict[str, Any]:
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
    um: dict[str, Any] = {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": (inp or 0) + (out or 0),
    }
    if details:
        um["input_token_details"] = details
    return {k: v for k, v in um.items() if v is not None}


def _start_root(client: Any, *, task_id: str, session_id: str, user_query: Any,
                metadata: dict[str, Any]) -> TraceState:
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
    except Exception as exc:  # noqa: BLE001 - fail open, but don't emit children
        logger.debug("LangSmith start_root failed: %s", type(exc).__name__)
        return TraceState(root_id=None, metadata=clean)
    return TraceState(root_id=root_id, metadata=clean)


def _record_turn(client: Any, state: TraceState, *, index: int, model: str | None,
                 thinking: str, text: str, usage: dict | None,
                 started_at: datetime, ended_at: datetime) -> None:
    if state.root_id is None:
        return
    turn_id = uuid.uuid4().hex
    out_msg: dict[str, Any] = {"role": "assistant", "content": text}
    if thinking:
        out_msg["thinking"] = _jsonable(thinking)
    extra: dict[str, Any] = {"metadata": {"model": model, "turn": index}}
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
    if state.root_id is None:
        return
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
    if state.root_id is None:
        return
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


def _finish(client: Any, state: TraceState, *, answer: str, num_turns: int | None,
            used_tools: list[str] | None, is_error: bool, error: str | None,
            session_id: str | None, total_cost_usd: float | None) -> None:
    if state.root_id is None:
        return
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


# ---------------------------------------------------------------------------
# Per-turn trace state, keyed so concurrent gateway turns never collide.
# _scope_prefix / _trace_key copied verbatim from the langfuse plugin.
# ---------------------------------------------------------------------------

# _STATE_LOCK guards dict membership of _TRACE_STATE and eviction only. Network
# I/O (create_run/update_run) is always done OUTSIDE the lock. Single-attribute
# writes to a TraceState (num_turns, answer, is_error, last_updated_at) rely on
# CPython's GIL for atomicity; on a free-threaded build they are benign races
# (last-writer-wins on independent scalars, never a corrupt tree).
_STATE_LOCK = threading.Lock()
_TRACE_STATE: dict[str, TraceState] = {}
# Bound the leak from turns that never reach a clean finish (interrupted,
# tool-only final step, empty final content); evict least-recently-updated.
_MAX_TRACE_STATE = 256


def _scope_prefix(task_id: str, session_id: str) -> str:
    if task_id:
        return f"task:{task_id}"
    if session_id:
        return f"session:{session_id}"
    return f"thread:{threading.get_ident()}"


def _trace_key(task_id: str, session_id: str, *, turn_id: str = "",
               api_request_id: str = "") -> str:
    if turn_id:
        return f"{_scope_prefix(task_id, session_id)}:turn:{turn_id}"
    if api_request_id:
        return f"{_scope_prefix(task_id, session_id)}:api:{api_request_id}"
    if task_id:
        return task_id
    return _scope_prefix(task_id, session_id)


def _extract_last_user_message(messages: Any) -> Any:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content")
    return None


def _pop_stale_locked() -> list[TraceState]:
    """Pop least-recently-updated trace state to make room for one new entry.

    Caller MUST hold ``_STATE_LOCK``. Evicts down to ``_MAX_TRACE_STATE - 1`` so
    the about-to-be-added entry leaves the dict at the ceiling. Returns the
    evicted states so the caller can close their root runs OUTSIDE the lock -
    no network I/O is done while the lock is held.
    """
    over = len(_TRACE_STATE) - (_MAX_TRACE_STATE - 1)
    if over <= 0:
        return []
    stale = sorted(_TRACE_STATE.items(), key=lambda kv: kv[1].last_updated_at)[:over]
    for key, _state in stale:
        _TRACE_STATE.pop(key, None)
    return [s for _, s in stale]


def _close_root(client: Any, state: TraceState) -> None:
    """Close a root run's end_time (evicted straggler or a lost-race duplicate)."""
    if client is None or state.root_id is None:
        return
    try:
        client.update_run(run_id=state.root_id, end_time=datetime.now(UTC))
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith close_root failed: %s", type(exc).__name__)


def _get_state(task_key: str) -> TraceState | None:
    with _STATE_LOCK:
        return _TRACE_STATE.get(task_key)


def _assistant_has_tool_calls(message: Any) -> bool:
    return bool(getattr(message, "tool_calls", None))


# ---------------------------------------------------------------------------
# Hook handlers. Registered for both the request-scoped (pre/post_api_request)
# and legacy turn-scoped (pre/post_llm_call) hook names, matching langfuse.
# ---------------------------------------------------------------------------

def on_pre_llm_request(*, task_id: str = "", session_id: str = "", platform: str = "",
                       model: str = "", provider: str = "", api_mode: str = "",
                       api_call_count: int = 0, request_messages: Any = None,
                       messages: Any = None, conversation_history: Any = None,
                       user_message: Any = None, turn_id: str = "",
                       api_request_id: str = "", **_: Any) -> None:
    client = _client()
    if client is None:
        return
    msgs = None
    for cand in (request_messages, messages, conversation_history):
        if isinstance(cand, list):
            msgs = cand
            break
    # Current Hermes also fires a turn-scoped pre_llm_call for context injection
    # (no messages list); tracing that would create an orphan root. Only open a
    # root when we have request messages or an explicit user_message.
    if msgs is None and user_message is None:
        return
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        existing = _TRACE_STATE.get(task_key)
        if existing is not None:
            existing.last_updated_at = time.time()
            return
        evicted = _pop_stale_locked()
    # Close evicted roots outside the lock so slow I/O never stalls other hooks.
    for st in evicted:
        _close_root(client, st)
    try:
        from hermes_cli.build_info import get_build_sha

        git_commit = get_build_sha()
    except Exception:  # noqa: BLE001
        git_commit = None
    metadata = {
        "source": "hermes", "task_id": task_id, "session_id": session_id,
        "turn_id": turn_id, "platform": platform, "provider": provider,
        "model": model, "api_mode": api_mode, "git_commit": git_commit,
    }
    state = _start_root(
        client, task_id=task_id, session_id=session_id,
        user_query=_extract_last_user_message(msgs) or user_message or "",
        metadata=metadata,
    )
    # Atomic set-if-absent: if a concurrent hook for the same key won the race,
    # keep its state and close our duplicate root (never orphan it).
    with _STATE_LOCK:
        winner = _TRACE_STATE.setdefault(task_key, state)
    if winner is not state:
        _close_root(client, state)


def on_post_llm_call(*, task_id: str = "", session_id: str = "", model: str = "",
                     provider: str = "", api_mode: str = "", api_call_count: int = 0,
                     assistant_message: Any = None, assistant_response: Any = None,
                     usage: Any = None, finish_reason: str = "",
                     assistant_tool_call_count: int = 0, started_at: Any = None,
                     ended_at: Any = None, api_duration: float = 0.0,
                     turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    client = _client()
    if client is None:
        return
    task_key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    state = _get_state(task_key)
    if state is None:
        # Legacy paths may skip the pre-hook; open the root lazily.
        on_pre_llm_request(task_id=task_id, session_id=session_id, model=model,
                           provider=provider, api_mode=api_mode,
                           api_call_count=api_call_count, user_message="",
                           turn_id=turn_id, api_request_id=api_request_id)
        state = _get_state(task_key)
        if state is None:
            return
    text = ""
    thinking = ""
    if assistant_message is not None:
        text = getattr(assistant_message, "content", "") or ""
        thinking = getattr(assistant_message, "reasoning", "") or ""
    elif isinstance(assistant_response, str):
        text = assistant_response
    end = ended_at if isinstance(ended_at, datetime) else datetime.now(UTC)
    start = started_at if isinstance(started_at, datetime) else end
    usage_dict = usage if isinstance(usage, dict) else None
    _record_turn(client, state, index=api_call_count or (state.num_turns + 1),
                 model=model, thinking=thinking, text=text, usage=usage_dict,
                 started_at=start, ended_at=end)
    if text:
        state.answer = text
    state.last_updated_at = time.time()
    has_tools = (
        _assistant_has_tool_calls(assistant_message)
        if assistant_message is not None
        else (assistant_tool_call_count > 0)
    )
    if text and not has_tools:
        with _STATE_LOCK:
            _TRACE_STATE.pop(task_key, None)
        _finish(client, state, answer=state.answer, num_turns=state.num_turns,
                used_tools=sorted(state.used_tools), is_error=state.is_error,
                error=state.error, session_id=session_id, total_cost_usd=None)


def on_pre_tool_call(*, tool_name: str = "", args: Any = None, task_id: str = "",
                     session_id: str = "", tool_call_id: str = "", turn_id: str = "",
                     api_request_id: str = "", **_: Any) -> None:
    client = _client()
    if client is None:
        return
    state = _get_state(_trace_key(task_id, session_id, turn_id=turn_id,
                                  api_request_id=api_request_id))
    if state is None:
        return
    _start_tool(client, state, tool_name=tool_name, args=args,
                tool_call_id=tool_call_id, started_at=datetime.now(UTC))


def on_post_tool_call(*, tool_name: str = "", args: Any = None, result: Any = None,
                      task_id: str = "", session_id: str = "", tool_call_id: str = "",
                      turn_id: str = "", api_request_id: str = "", status: str = "",
                      error_type: str = "", **_: Any) -> None:
    client = _client()
    if client is None:
        return
    state = _get_state(_trace_key(task_id, session_id, turn_id=turn_id,
                                  api_request_id=api_request_id))
    if state is None:
        return
    is_error = bool(error_type) or (status not in ("", "ok", "success"))
    _end_tool(client, state, tool_name=tool_name, result=result,
              tool_call_id=tool_call_id, is_error=is_error, ended_at=datetime.now(UTC))


def on_api_request_error(*, task_id: str = "", session_id: str = "", error: Any = None,
                         turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    if _client() is None:
        return
    state = _get_state(_trace_key(task_id, session_id, turn_id=turn_id,
                                  api_request_id=api_request_id))
    if state is None:
        return
    state.is_error = True
    if error is not None:
        state.error = str(error)


def register(ctx: Any) -> None:
    # pre/post_api_request fire per API call (preferred); pre/post_llm_call are
    # the legacy per-turn names. Register both so the plugin works across
    # Hermes versions, matching the langfuse plugin.
    ctx.register_hook("pre_api_request", on_pre_llm_request)
    ctx.register_hook("post_api_request", on_post_llm_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_request)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("api_request_error", on_api_request_error)
