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
