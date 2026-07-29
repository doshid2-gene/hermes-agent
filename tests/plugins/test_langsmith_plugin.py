"""Tests for the bundled observability/langsmith plugin.

Emits the deep-claude-agent ``TaskTrace`` run-tree (chain -> llm "turn N" -> tool)
to LangSmith. These tests use a spy client (no SDK, no network) for the tree/usage/
finish logic, plus one importorskip test for the real client's endpoint/SSL wiring.
"""
from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone

import pytest


def _fresh_module():
    """Import the plugin fresh so its cached client / trace state reset."""
    sys.modules.pop("plugins.observability.langsmith", None)
    return importlib.import_module("plugins.observability.langsmith")


def _now():
    return datetime.now(timezone.utc)


class _SpyClient:
    """Records create_run / update_run calls instead of hitting LangSmith."""

    def __init__(self):
        self.created = []
        self.updated = []

    def create_run(self, **kw):
        self.created.append(kw)

    def update_run(self, **kw):
        self.updated.append(kw)


def _by_name(spy, name):
    for c in spy.created:
        if c.get("name") == name:
            return c
    raise AssertionError(
        f"no create_run named {name!r}; got {[c.get('name') for c in spy.created]}"
    )


# ---------------------------------------------------------------------------
# Task 1: client builder + silent no-op gate
# ---------------------------------------------------------------------------

def test_disabled_without_key_returns_no_client(monkeypatch):
    for var in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    ls = _fresh_module()
    assert ls.tracing_enabled() is False
    assert ls._client() is None


def test_client_uses_endpoint_verbatim_and_verify_false(monkeypatch):
    pytest.importorskip("langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv(
        "LANGSMITH_ENDPOINT", "https://langsmith.dev.lightship.gene.com/api/v1"
    )
    captured = {}

    class _FakeSession:
        def __init__(self):
            self.verify = True

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session = _FakeSession()

    warned = {"called": False}
    import urllib3
    monkeypatch.setattr(
        urllib3, "disable_warnings", lambda *a, **k: warned.__setitem__("called", True)
    )
    import langsmith
    monkeypatch.setattr(langsmith, "Client", _FakeClient)

    ls = _fresh_module()
    client = ls._client()
    assert client is not None
    assert captured["api_url"] == "https://langsmith.dev.lightship.gene.com/api/v1"
    assert captured["api_key"] == "ls-test-key"
    assert client.session.verify is False
    assert warned["called"] is True


# ---------------------------------------------------------------------------
# Task 2: TaskTrace-shaped tree helpers (chain -> llm -> tool, usage, finish)
# ---------------------------------------------------------------------------

def test_usage_metadata_shape_matches_tasktrace():
    ls = _fresh_module()
    um = ls._usage_metadata({
        "input_tokens": 3000, "output_tokens": 800,
        "cache_read_tokens": 2000, "cache_write_tokens": 500,
    })
    assert um == {
        "input_tokens": 3000, "output_tokens": 800, "total_tokens": 3800,
        "input_token_details": {"cache_read": 2000, "cache_creation": 500},
    }


def test_usage_metadata_empty_is_empty():
    ls = _fresh_module()
    assert ls._usage_metadata(None) == {}
    # no cache tokens -> no input_token_details key
    assert ls._usage_metadata({"input_tokens": 5, "output_tokens": 7}) == {
        "input_tokens": 5, "output_tokens": 7, "total_tokens": 12,
    }


def test_llm_turn_nests_under_root_chain():
    ls = _fresh_module()
    spy = _SpyClient()
    state = ls._start_root(spy, task_id="t1", session_id="s1", user_query="hi",
                           metadata={"config": "hermes"})
    ls._record_turn(spy, state, index=1, model="m", thinking="th", text="hello",
                    usage=None, started_at=_now(), ended_at=_now())
    root = _by_name(spy, "task:t1")
    turn = _by_name(spy, "turn 1")
    assert root["run_type"] == "chain"
    assert root["inputs"] == {"question": "hi"}
    assert turn["run_type"] == "llm"
    assert turn["parent_run_id"] == state.root_id


def test_tool_nests_under_its_turn():
    ls = _fresh_module()
    spy = _SpyClient()
    state = ls._start_root(spy, task_id="t1", session_id="s1", user_query="hi", metadata={})
    ls._record_turn(spy, state, index=1, model="m", thinking="", text="",
                    usage=None, started_at=_now(), ended_at=_now())
    ls._start_tool(spy, state, tool_name="read_file", args={"path": "a"},
                   tool_call_id="tc1", started_at=_now())
    ls._end_tool(spy, state, tool_name="read_file", result="ok",
                 tool_call_id="tc1", is_error=False, ended_at=_now())
    turn = _by_name(spy, "turn 1")
    tool = _by_name(spy, "read_file")
    assert tool["run_type"] == "tool"
    assert tool["parent_run_id"] == turn["id"]
    upd = [u for u in spy.updated if u.get("run_id") == tool["id"]]
    assert upd and upd[-1]["outputs"] == {"output": "ok"}
    assert "read_file" in state.used_tools


def test_finish_merges_lineage_and_metrics():
    ls = _fresh_module()
    spy = _SpyClient()
    state = ls._start_root(spy, task_id="t1", session_id="s1", user_query="hi",
                           metadata={"config": "hermes", "git_commit": "abc"})
    ls._finish(spy, state, answer="done", num_turns=2, used_tools=["read_file"],
               is_error=False, error=None, session_id="s1", total_cost_usd=0.01)
    upd = [u for u in spy.updated if u.get("run_id") == state.root_id]
    assert upd, "root must be updated on finish"
    last = upd[-1]
    assert last["outputs"] == {"answer": "done"}
    md = last["extra"]["metadata"]
    assert md["git_commit"] == "abc"   # start lineage preserved (merge)
    assert md["config"] == "hermes"
    assert md["num_turns"] == 2
    assert md["used_tools"] == ["read_file"]
    assert md["is_error"] is False
    assert md["total_cost_usd"] == 0.01


def test_error_marks_root():
    ls = _fresh_module()
    spy = _SpyClient()
    state = ls._start_root(spy, task_id="t1", session_id="s1", user_query="hi", metadata={})
    ls._finish(spy, state, answer="", num_turns=1, used_tools=[], is_error=True,
               error="boom", session_id="s1", total_cost_usd=None)
    last = [u for u in spy.updated if u.get("run_id") == state.root_id][-1]
    assert last["error"] == "boom"
