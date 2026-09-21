# mcp-open-code-bridge

An [MCP](https://modelcontextprotocol.io) (Model Context Protocol) server that exposes [OpenCode](https://opencode.ai) coding agents as MCP tools, running each one inside an isolated, disposable Docker sandbox.

## Overview

Each configured OpenCode agent (`implementer`, `reviewer`, `tester`, ...) is discovered from `.md` files with YAML frontmatter and registered as its own MCP tool. When a tool is called, the server:

1. Spins up a dedicated Docker sandbox for the session via the [`sbx`](https://github.com/) (Docker Sandboxes) CLI.
2. Starts `opencode serve` inside that sandbox and waits for it to come up.
3. Talks to it over HTTP, posting the prompt and racing the response against the sandbox's own SSE `/event` stream so a pending permission request (e.g. "may I run this shell command?") surfaces immediately instead of hanging.
4. Lets the caller resolve any pending permission via the `answer_permission` tool, then resumes the run.
5. Tears the sandbox down once the run completes (with `atexit`/`SIGTERM`/`SIGINT` safety nets to guarantee cleanup even on a crash).

The sandboxing/session engine is shared code, and there are two thin front-ends on top of it:

- **`server/orca.py`** — integrates with [Orca](https://github.com/): opens a visible terminal pane per session (`orca terminal create` + `opencode attach`) so a human can watch/join live, and forwards busy/waiting/idle status to Orca's local hook HTTP server.
- **`server/generic.py`** — the same engine with no integration wired in, for use without Orca.

## Architecture

```
MCP client (Claude Code, etc.)
        │
        ▼
server/orca.py  or  server/generic.py   (MCP entry point, tool registration)
        │
        ▼
core/package.py                            (session lifecycle, sandbox management,
        │                                    agent discovery, permission handling)
        ▼
sbx sandbox  ──►  opencode serve            (HTTP + SSE API, one sandbox per session)
```

## Directory structure

```
.
├── core/
│   └── package.py       # shared engine: sandbox lifecycle, agent discovery, prompt execution, permissions
├── server/
│   ├── orca.py           # MCP entry point with Orca terminal + status-hook integration
│   └── generic.py        # MCP entry point with no integration
├── test/
│   ├── run_tests.py       # aggregates and runs all test modules below
│   ├── test_package.py    # engine unit tests
│   ├── test_orca.py       # Orca hook tests
│   └── test_generic.py    # no-op wiring tests
└── logs/
    └── orca_bridge.log    # rotating log file (created at runtime)
```

## Prerequisites

- Python 3.10+ (developed against 3.14).
- [`sbx`](https://github.com/) — Docker Sandboxes CLI, used to create/exec/list/remove sandboxes.
- [`opencode`](https://opencode.ai) — the OpenCode agent CLI, run inside each sandbox.
- [`orca`](https://github.com/) — only required for `server/orca.py`'s terminal + status-hook integration; not needed for `server/generic.py`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the two packages the engine imports directly (`mcp`, `PyYAML`); `mcp` pulls in its own transitive dependencies (`httpx2`, `pydantic`, `starlette`, `uvicorn`, `sse-starlette`, ...) automatically.

## Configuration

### Agent definitions

Agents are discovered from `.md` files with YAML frontmatter, read from:

- `~/.config/opencode/agent/*.md` — global agents
- `<project_dir>/.opencode/agent/*.md` — project-local agents (override global ones with the same name)

Each file's name (minus `.md`) becomes the MCP tool name. Example (`implementer.md`):

```yaml
---
description: Implements an approved technical plan
mode: primary
model: opencode/big-pickle
permission:
  read: allow
  edit: allow
  bash: deny
  external_directory: deny
network: balanced
---
```

`network` controls sandbox network access: `balanced` (default) allows it; `none` passes `--deny-network **` to `sbx create` (note: this also blocks the agent's own model API calls, so it effectively disables the agent).

### Environment variables

| Variable | Purpose |
|---|---|
| `CLAUDE_PROJECT_DIR` | Project directory root (falls back to the current working directory); used to locate `.opencode/agent/`. |
| `OPENCODE_CONFIG_DIR` | Forwarded into the sandbox environment and bind-mounted read-only into `sbx create`. |
| `ORCA_AGENT_HOOK_ENDPOINT` | Path to a file with `ORCA_AGENT_HOOK_PORT`/`TOKEN`/`ENV`/`VERSION` lines describing Orca's hook server (preferred over the individual vars below). |
| `ORCA_AGENT_HOOK_PORT`, `ORCA_AGENT_HOOK_TOKEN`, `ORCA_AGENT_HOOK_ENV`, `ORCA_AGENT_HOOK_VERSION` | Fallback coordinates for Orca's local hook HTTP server, used to report session status. |
| `ORCA_AGENT_LAUNCH_TOKEN`, `ORCA_WORKTREE_ID` | Forwarded into the sandbox and included in status POST bodies. |
| `ORCA_PANE_KEY`, `ORCA_TAB_ID`, `ORCA_TERMINAL_HANDLE` | Set as sandbox environment variables by the Orca visibility hook. |

The `ORCA_*` variables only matter when using `server/orca.py`.

## Running the server

Register the server with your MCP client, pointing at either front-end. Example `.mcp.json`:

```json
{
  "mcpServers": {
    "mcp-open-code-bridge": {
      "type": "stdio",
      "command": "/path/to/mcp-open-code-bridge/.venv/bin/python3",
      "args": ["/path/to/mcp-open-code-bridge/server/orca.py"],
      "env": {}
    }
  }
}
```

Use `server/generic.py` instead of `server/orca.py` if you don't want the Orca integration.

## Available MCP tools

- One tool per discovered agent (e.g. `implementer`, `reviewer`, `tester`), named after the agent's `.md` file stem — invokes that agent's prompt in a fresh sandbox session.
- `answer_permission` — resolves a pending permission request (`once` / `always` / `reject`) raised during a run, then resumes it.

## Testing

```bash
python test/run_tests.py
```

This runs `test_package`, `test_orca`, and `test_generic` (`unittest`-based, using `IsolatedAsyncioTestCase` and mocks for the sandbox/HTTP layer — no real Docker sandbox or `opencode` install required).

## Development notes

- Every session gets its own sandbox and its own `opencode serve` process; sandboxes are torn down after each run, plus on process exit and on `SIGTERM`/`SIGINT`, so no sandbox should be left running if the server dies unexpectedly.
- Transient connection errors during a run trigger one retry, evicting and recreating the sandbox backend before giving up.
- Because a sandbox's network policy blocks it from reaching the host's own loopback address, status reporting to Orca (busy/waiting/idle) is done by the MCP server itself, on the sandbox's behalf, rather than from inside the sandbox.
