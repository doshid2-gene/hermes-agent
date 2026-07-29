# LangSmith Observability Plugin

This plugin ships bundled with Hermes but is **opt-in** - it only loads when you
explicitly enable it.

It emits the same LangSmith run-tree the deep-claude-agent baseline produces
(`TaskTrace`), so Hermes runs can be compared side-by-side in the **same LangSmith
project**:

```
task:<id>              chain   (the whole turn; git commit + model in metadata)
 |- turn 1            llm     (assistant thinking + text + token usage)
 |   |- <tool>        tool    (input=args, output=result)
 |- turn 2            llm
 ...
```

## Enable

```bash
pip install langsmith
hermes plugins enable observability/langsmith
```

## Required credentials

Set these in `~/.hermes/.env` (or export them in the shell). Use the same values
as the baseline so runs land together:

```bash
LANGSMITH_API_KEY=...                                            # or LANGCHAIN_API_KEY
LANGSMITH_ENDPOINT=https://langsmith.dev.lightship.gene.com/api/v1   # optional; self-hosted
LANGSMITH_PROJECT=claude-data-agent                              # optional; default: hermes-agent
```

Notes for the Roche self-hosted server:
- The endpoint is used **verbatim** as the client `api_url` - keep the `/api/v1`
  suffix.
- SSL verification is **disabled** (`session.verify = False`) and insecure-request
  warnings are silenced, matching the baseline's testing conventions.

Without the `langsmith` SDK or an API key, every hook is a silent no-op - the plugin
fails open and never affects the run.

## Verify

```bash
hermes plugins list                 # observability/langsmith should show "enabled"
hermes chat -q "hello"              # then look for a "task:*" trace in your project
```

## How it works

The plugin subscribes to Hermes lifecycle hooks (`pre/post_api_request`,
`pre/post_tool_call`, `api_request_error`, plus the legacy `pre/post_llm_call`) and
maps them onto LangSmith runs via `langsmith.Client.create_run` / `update_run`. Token
usage is normalized through `agent.usage_pricing` into LangSmith's `usage_metadata`
shape. Per-turn state is keyed by task/session/turn id so concurrent gateway turns
never collide, and stale state is evicted to bound memory.

`delegate_task` subagents appear as ordinary `tool` runs under the turn that spawned
them.
