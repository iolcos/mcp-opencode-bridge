import asyncio
import atexit
import json
import logging
import os
import re
import shlex
import signal
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import httpx2
import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


mcp = MCPServer("orca-bridge")

LOG_PATH = Path(__file__).resolve().parent / "logs" / "orca_bridge.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("orca_bridge")
logger.setLevel(logging.INFO)
logger.propagate = False

_log_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(_log_handler)


def get_project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())).resolve()


# Not called anywhere yet: kept for the future review_browser implementation
# (see the stub below), not accidental dead code.
def load_browser_config() -> dict[str, Any]:
    project_dir = get_project_dir()
    config_file = project_dir / "browser-review.yaml"

    if not config_file.exists():
        raise RuntimeError(f"Browser config not found: {config_file}")

    try:
        with config_file.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read browser config: {config_file}"
        ) from exc

    if not isinstance(config, dict):
        raise RuntimeError("Browser config must contain a YAML object")

    return config


def get_browser_account(
    config: dict[str, Any],
    account_name: str,
) -> dict[str, str]:
    accounts = config.get("accounts")

    if not isinstance(accounts, dict):
        raise RuntimeError("browser-review.yaml: 'accounts' must be an object")

    account = accounts.get(account_name)

    if not isinstance(account, dict):
        raise RuntimeError(
            f"Unknown browser account: {account_name}"
        )

    username = account.get("username")
    password = account.get("password")

    if not username or not password:
        raise RuntimeError(
            f"Browser account '{account_name}' must define username and password"
        )

    return {
        "username": str(username),
        "password": str(password),
    }


ORCA_COMMAND_TIMEOUT_SECONDS = 30


async def run_command(
    command: list[str],
    cwd: str | None = None,
    timeout: float | None = None,
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        start_new_session=True,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        # start_new_session puts the process in its own process group, so
        # killing the group (not just the direct child) also reaches any
        # descendants it may have spawned.
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        raise RuntimeError(
            f"Command timed out after {timeout}s: {shlex.join(command)}"
        )

    return (
        process.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


_json_decoder = json.JSONDecoder()


def parse_json_output(output: str) -> dict[str, Any]:
    """
    Parse the JSON object returned by an Orca CLI command.

    Orca normally returns one JSON object, but it may be pretty-printed
    across multiple lines, so the whole output is tried as a single JSON
    document first (the common, fast case). If that fails -- e.g. a
    banner/log line before or after the JSON -- fall back to scanning for
    the first "{" that starts a fully valid JSON document, which handles
    single-line JSON, pretty-printed multi-line JSON, and surrounding
    junk uniformly.
    """
    stripped = output.strip()

    if stripped:
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, dict):
                return value

    for index, char in enumerate(output):
        if char != "{":
            continue

        try:
            value, _ = _json_decoder.raw_decode(output, index)
        except json.JSONDecodeError:
            continue

        if isinstance(value, dict):
            return value

    raise RuntimeError(
        f"Could not parse JSON output:\n{output}"
    )


def extract_orca_terminal_handle(result: dict[str, Any]) -> str:
    inner_result = result.get("result")
    terminal = inner_result.get("terminal") if isinstance(inner_result, dict) else None
    handle = terminal.get("handle") if isinstance(terminal, dict) else None

    if not handle:
        raise RuntimeError(
            "Orca did not return a terminal handle:\n"
            f"{json.dumps(result, indent=2)}"
        )

    return str(handle)


# ---------------------------------------------------------------------------
# OpenCode server management.
#
# Architecture:
#
#     MCP  --HTTP/JSON-->  opencode serve  (data channel, drives the run)
#     MCP  --orca CLI-->   Orca terminal running `opencode attach` (visibility
#                          channel only -- never read back by the MCP)
#
# The MCP never scrapes terminal output for opencode's own responses anymore:
# `opencode serve` gives structured JSON directly.
# ---------------------------------------------------------------------------

OPENCODE_SERVE_STARTUP_TIMEOUT_SECONDS = 15
OPENCODE_PROMPT_TIMEOUT_SECONDS = 900

_LISTENING_URL_RE = re.compile(r"listening on (http://\S+)")

# project_dir -> (process, base_url)
_opencode_servers: dict[Path, tuple[asyncio.subprocess.Process, str]] = {}
_opencode_servers_lock = asyncio.Lock()

# Keeps references to fire-and-forget background tasks (stdout drains) alive
# for as long as the process runs -- otherwise they could be garbage
# collected mid-flight.
_background_tasks: set[asyncio.Task] = set()

# session_id -> Orca terminal handle showing the live `opencode attach` view.
_session_terminals: dict[str, str] = {}


def _spawn_background_task(coro: Any) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _drain_stream(stream: asyncio.StreamReader) -> None:
    """Keep consuming a subprocess pipe so it never fills up and blocks the child."""
    while True:
        line = await stream.readline()
        if not line:
            return


def _kill_opencode_servers() -> None:
    """Best-effort synchronous cleanup, registered with atexit."""
    for process, _base_url in _opencode_servers.values():
        if process.returncode is not None:
            continue
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            logger.exception("Failed to terminate opencode serve process %s", process.pid)


atexit.register(_kill_opencode_servers)


async def _start_opencode_server(project_dir: Path) -> tuple[asyncio.subprocess.Process, str]:
    process = await asyncio.create_subprocess_exec(
        "opencode",
        "serve",
        "--port",
        "0",
        "--hostname",
        "127.0.0.1",
        cwd=str(project_dir),
        stdout=asyncio.subprocess.PIPE,
        # Merged into stdout: we don't know in advance which stream carries
        # the startup banner, and only one stream needs draining afterwards.
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )

    async def read_until_listening() -> str:
        while True:
            line = await process.stdout.readline()

            if not line:
                raise RuntimeError(
                    "opencode serve exited before printing a listening URL "
                    f"(exit code {process.returncode})"
                )

            match = _LISTENING_URL_RE.search(line.decode(errors="replace"))

            if match:
                return match.group(1)

    try:
        base_url = await asyncio.wait_for(
            read_until_listening(), timeout=OPENCODE_SERVE_STARTUP_TIMEOUT_SECONDS
        )
    except (asyncio.TimeoutError, RuntimeError):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        raise

    _spawn_background_task(_drain_stream(process.stdout))

    return process, base_url


async def ensure_opencode_server(project_dir: Path) -> str:
    """Start `opencode serve` for this project if needed, reuse it otherwise."""
    async with _opencode_servers_lock:
        cached = _opencode_servers.get(project_dir)

        if cached is not None:
            process, base_url = cached
            if process.returncode is None:
                return base_url
            del _opencode_servers[project_dir]

        process, base_url = await _start_opencode_server(project_dir)
        _opencode_servers[project_dir] = (process, base_url)
        return base_url


async def create_opencode_session(base_url: str, title: str) -> str:
    async with httpx2.AsyncClient(timeout=ORCA_COMMAND_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{base_url}/session", json={"title": title})

    if response.is_error:
        raise RuntimeError(
            f"Failed to create OpenCode session:\n{response.text}"
        )

    session_id = response.json().get("id")

    if not session_id:
        raise RuntimeError(
            f"OpenCode did not return a session id:\n{response.text}"
        )

    return str(session_id)


async def send_opencode_prompt(
    base_url: str,
    session_id: str,
    agent: str,
    prompt: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    async with httpx2.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"{base_url}/session/{session_id}/message",
            json={
                "agent": agent,
                "parts": [{"type": "text", "text": prompt}],
            },
        )

    if response.is_error:
        raise RuntimeError(
            f"OpenCode prompt failed (status {response.status_code}):\n{response.text}"
        )

    return response.json()


async def ensure_visible_terminal(
    base_url: str,
    session_id: str,
    agent: str,
    project_dir: Path,
) -> None:
    """
    Best-effort: open an Orca terminal running `opencode attach`, purely so
    the session is visible/watchable in Orca. This is never read back by the
    MCP, and a failure here must never fail the actual implement/review call.
    """
    try:
        handle = _session_terminals.get(session_id)

        if handle is not None:
            returncode, _, _ = await run_command(
                ["orca", "terminal", "show", "--terminal", handle, "--json"],
                timeout=ORCA_COMMAND_TIMEOUT_SECONDS,
            )
            if returncode == 0:
                return

        returncode, stdout, stderr = await run_command(
            [
                "orca",
                "terminal",
                "create",
                "--worktree",
                "current",
                "--title",
                f"OpenCode {agent}",
                "--command",
                f"opencode attach {base_url} --session {session_id}",
                "--json",
            ],
            cwd=str(project_dir),
            timeout=ORCA_COMMAND_TIMEOUT_SECONDS,
        )

        if returncode != 0:
            raise RuntimeError(stderr or stdout)

        _session_terminals[session_id] = extract_orca_terminal_handle(
            parse_json_output(stdout)
        )
    except Exception:
        logger.exception(
            "Failed to open/verify the visibility terminal for session %s", session_id
        )


async def run_opencode(
    agent: str,
    prompt: str,
    session_id: str | None = None,
    timeout_seconds: int = OPENCODE_PROMPT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    project_dir = get_project_dir()
    base_url = await ensure_opencode_server(project_dir)

    if session_id is None:
        session_id = await create_opencode_session(base_url, title=f"orca-bridge {agent}")

    await ensure_visible_terminal(base_url, session_id, agent, project_dir)

    result = await send_opencode_prompt(
        base_url, session_id, agent, prompt, timeout_seconds
    )

    info = result.get("info")
    info = info if isinstance(info, dict) else {}

    error = info.get("error")

    if error:
        raise RuntimeError(
            "OpenCode returned an error:\n"
            f"{json.dumps(error, indent=2)}\n"
            f"session_id={session_id}"
        )

    parts = result.get("parts")
    parts = parts if isinstance(parts, list) else []

    text = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and part.get("type") == "text"
    )

    return {
        "session_id": session_id,
        "response": text,
        "finish_reason": info.get("finish"),
    }


@mcp.tool()
async def implement(
    plan: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    try:
        return await run_opencode(
            agent="implementer",
            session_id=session_id,
            prompt=f"""
You are the implementation agent.

An architect has approved the following implementation plan:

--- PLAN ---

{plan}

--- END PLAN ---

Implement this plan in the current repository.

Rules:
- Inspect the existing code before making changes.
- Follow the plan.
- Follow the project CLAUDE.md.
- Do not redesign the architecture.
- Do not make unrelated improvements.
- Do not read the .env file.
- Do not run artisan, migrations, composer or tests.
- Instead, report the commands that should be executed by the user.
- Fix implementation errors you encounter that do not require running commands yourself.

At the end report:
- files changed
- implementation details
- commands the user should execute
- remaining issues
- uncertainties
""",
        )
    except Exception as exc:
        logger.exception("implement() failed")
        raise ToolError(str(exc)) from exc


@mcp.tool()
async def review(
    plan: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    try:
        return await run_opencode(
            agent="reviewer",
            session_id=session_id,
            prompt=f"""
You are the review agent.

Review the current implementation against this approved plan:

--- PLAN ---

{plan}

--- END PLAN ---

Rules:
- Do not modify files.
- Follow the project CLAUDE.md.
- Do not read the .env file.
- Do not run project commands.
- Do not modify configuration.
- Do not fix problems yourself.
- Do not perform browser testing.

Check:
- correctness
- adherence to the plan
- architecture
- regressions
- edge cases
- security
- maintainability
- tests
- project conventions

Return:
## BLOCKING
...
## NON_BLOCKING
...
## MISSING_TESTS
...
## MISSING_BROWSER_VERIFICATION
...
## POSITIVE
...
## VERDICT
APPROVE | REQUEST_CHANGES
""",
        )
    except Exception as exc:
        logger.exception("review() failed")
        raise ToolError(str(exc)) from exc


@mcp.tool()
async def review_browser(
    test_plan: str,
    account: str,
) -> dict[str, Any]:
    # Keep the existing implementation of this tool here.
    # It is intentionally omitted from the Orca/OpenCode transport change.
    raise ToolError("review_browser is not implemented yet.")


if __name__ == "__main__":
    mcp.run()
