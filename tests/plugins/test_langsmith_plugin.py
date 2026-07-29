"""Tests for the bundled observability/langsmith plugin.

Emits the deep-claude-agent ``TaskTrace`` run-tree (chain -> llm "turn N" -> tool)
to LangSmith. These tests use a spy client (no SDK, no network) for the tree/usage/
finish logic, plus one importorskip test for the real client's endpoint/SSL wiring.
"""
from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "langsmith"


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


# ---------------------------------------------------------------------------
# Task 3: hook handlers building the tree via the plugin system
# ---------------------------------------------------------------------------

def _module_with_spy(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    ls = _fresh_module()
    spy = _SpyClient()
    monkeypatch.setattr(ls, "_client", lambda: spy)
    return ls, spy


def test_hooks_build_chain_llm_tool_and_finish(monkeypatch):
    ls, spy = _module_with_spy(monkeypatch)
    kw = dict(task_id="t1", session_id="s1", turn_id="turn-abc", api_request_id="req1")
    ls.on_pre_llm_request(model="m", provider="openai",
                          messages=[{"role": "user", "content": "hi"}],
                          api_call_count=1, **kw)

    class _Msg:
        content = ""
        reasoning = "thinking..."
        tool_calls = [object()]

    ls.on_post_llm_call(model="m", provider="openai", api_call_count=1,
                        assistant_message=_Msg(),
                        usage={"input_tokens": 10, "output_tokens": 5},
                        finish_reason="tool_calls", assistant_tool_call_count=1, **kw)
    ls.on_pre_tool_call(tool_name="read_file", args={"path": "a"}, tool_call_id="tc1", **kw)
    ls.on_post_tool_call(tool_name="read_file", args={"path": "a"}, result="ok",
                         tool_call_id="tc1", **kw)

    class _Final:
        content = "the answer"
        reasoning = ""
        tool_calls = []

    ls.on_post_llm_call(model="m", provider="openai", api_call_count=2,
                        assistant_message=_Final(),
                        usage={"input_tokens": 8, "output_tokens": 12},
                        finish_reason="stop", assistant_tool_call_count=0, **kw)

    root = _by_name(spy, "task:t1")
    turn1 = _by_name(spy, "turn 1")
    tool = _by_name(spy, "read_file")
    assert root["run_type"] == "chain"
    assert turn1["parent_run_id"] == root["id"]
    assert tool["parent_run_id"] == turn1["id"]
    root_updates = [u for u in spy.updated if u.get("run_id") == root["id"]]
    assert root_updates and root_updates[-1]["outputs"]["answer"] == "the answer"
    assert root_updates[-1]["extra"]["metadata"]["used_tools"] == ["read_file"]
    assert root_updates[-1]["extra"]["metadata"]["num_turns"] == 2


def test_api_request_error_marks_root(monkeypatch):
    ls, spy = _module_with_spy(monkeypatch)
    kw = dict(task_id="t2", session_id="s2", turn_id="turn-x", api_request_id="r1")
    ls.on_pre_llm_request(model="m", provider="openai",
                          messages=[{"role": "user", "content": "hi"}],
                          api_call_count=1, **kw)
    ls.on_api_request_error(error="ratelimited", **kw)

    class _Final:
        content = "recovered"
        reasoning = ""
        tool_calls = []

    ls.on_post_llm_call(model="m", provider="openai", api_call_count=1,
                        assistant_message=_Final(), usage=None,
                        finish_reason="stop", assistant_tool_call_count=0, **kw)
    root = _by_name(spy, "task:t2")
    last = [u for u in spy.updated if u.get("run_id") == root["id"]][-1]
    assert last["error"] == "ratelimited"


def test_post_without_pre_lazily_creates_root(monkeypatch):
    ls, spy = _module_with_spy(monkeypatch)
    kw = dict(task_id="t3", session_id="s3", turn_id="turn-y", api_request_id="r2")

    class _Final:
        content = "hello"
        reasoning = ""
        tool_calls = []

    ls.on_post_llm_call(model="m", provider="openai", api_call_count=1,
                        assistant_message=_Final(), usage=None,
                        finish_reason="stop", assistant_tool_call_count=0, **kw)
    assert _by_name(spy, "task:t3")["run_type"] == "chain"
    assert _by_name(spy, "turn 1")["run_type"] == "llm"


def test_disabled_hooks_never_raise(monkeypatch):
    for var in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    ls = _fresh_module()
    # no client -> every hook is a no-op and must not raise
    ls.on_pre_llm_request(task_id="t", session_id="s",
                          messages=[{"role": "user", "content": "x"}])
    ls.on_post_llm_call(task_id="t", session_id="s", assistant_message=None, usage=None)
    ls.on_pre_tool_call(task_id="t", session_id="s", tool_name="x", args={}, tool_call_id="1")
    ls.on_post_tool_call(task_id="t", session_id="s", tool_name="x", result="y", tool_call_id="1")
    ls.on_api_request_error(task_id="t", session_id="s", error="e")


def test_register_binds_expected_hooks(monkeypatch):
    ls = _fresh_module()
    bound = {}

    class _Ctx:
        def register_hook(self, name, fn):
            bound[name] = fn

    ls.register(_Ctx())
    assert set(bound) == {
        "pre_api_request", "post_api_request",
        "pre_llm_call", "post_llm_call",
        "pre_tool_call", "post_tool_call",
        "api_request_error",
    }


def test_start_root_failure_is_noop():
    """If the root create_run fails, the state is a no-op: no orphaned child
    runs get created under a non-existent parent (matches TaskTrace)."""
    ls = _fresh_module()

    class _FailingClient:
        def __init__(self):
            self.created = []
            self.updated = []

        def create_run(self, **kw):
            raise RuntimeError("boom")

        def update_run(self, **kw):
            self.updated.append(kw)

    c = _FailingClient()
    state = ls._start_root(c, task_id="t", session_id="s", user_query="q", metadata={})
    assert state.root_id is None
    ls._record_turn(c, state, index=1, model="m", thinking="", text="hi",
                    usage=None, started_at=_now(), ended_at=_now())
    ls._start_tool(c, state, tool_name="x", args={}, tool_call_id="1", started_at=_now())
    ls._end_tool(c, state, tool_name="x", result="r", tool_call_id="1",
                 is_error=False, ended_at=_now())
    ls._finish(c, state, answer="a", num_turns=1, used_tools=[], is_error=False,
               error=None, session_id="s", total_cost_usd=None)
    # No child runs and no updates against a parent that never existed.
    assert c.updated == []


def test_eviction_closes_oldest_root(monkeypatch):
    ls, spy = _module_with_spy(monkeypatch)
    monkeypatch.setattr(ls, "_MAX_TRACE_STATE", 2)
    for i in range(3):
        ls.on_pre_llm_request(task_id=f"t{i}", session_id="s", turn_id=f"turn{i}",
                              api_request_id=f"r{i}",
                              messages=[{"role": "user", "content": "hi"}])
    with ls._STATE_LOCK:
        assert len(ls._TRACE_STATE) <= 2
    root0 = _by_name(spy, "task:t0")
    closed = [u for u in spy.updated
              if u.get("run_id") == root0["id"] and "end_time" in u]
    assert closed, "evicted oldest root should be closed with an end_time"


# ---------------------------------------------------------------------------
# Task 4: manifest + layout + opt-in discovery
# ---------------------------------------------------------------------------

def test_plugin_directory_and_files_exist():
    assert PLUGIN_DIR.is_dir()
    assert (PLUGIN_DIR / "plugin.yaml").exists()
    assert (PLUGIN_DIR / "__init__.py").exists()
    assert (PLUGIN_DIR / "README.md").exists()


def test_manifest_fields():
    data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    assert data["name"] == "langsmith"
    assert data["version"]
    assert set(data["hooks"]) == {
        "pre_api_request", "post_api_request",
        "pre_llm_call", "post_llm_call",
        "pre_tool_call", "post_tool_call",
        "api_request_error",
    }
    assert "LANGSMITH_API_KEY" in data["requires_env"]


def test_plugin_is_discovered_as_standalone_opt_in(tmp_path, monkeypatch):
    """Scanner finds the plugin but does NOT load it by default (opt-in)."""
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    manager = plugins_mod.PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins.get("observability/langsmith")
    assert loaded is not None, "plugin not discovered"
    assert loaded.enabled is False
    assert "not enabled" in (loaded.error or "").lower()
