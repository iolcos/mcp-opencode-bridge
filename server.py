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

# session_id -> (process, base_url, terminal_handle_or_None). Each session
# gets its own dedicated opencode serve process (see _create_session_pane /
# _session_server_env below) so its busy/idle/waiting status can be attributed
# to its own Orca terminal instead of a shared server with no single owner.
_session_state: dict[str, tuple[asyncio.subprocess.Process, str, str | None]] = {}

# session_id -> lock, so two concurrent calls sharing a session_id don't each
# decide "no backend yet" and spawn a duplicate server/terminal. Lazily
# populated; safe to grow unboundedly given each entry is a bare Lock
# (negligible memory).
_session_locks: dict[str, asyncio.Lock] = {}


def _get_session_lock(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


# session_id -> queue of pending opencode permission.asked event payloads,
# fed by _watch_permission_events. Lazily populated so an event that arrives
# before anything is watching for it isn't lost.
_permission_queues: dict[str, asyncio.Queue] = {}

# session_id -> the send_opencode_prompt() task still running server-side
# after run_opencode() returned early with status "permission_required".
# Kept alive (never cancelled) so answer_permission() can pick its result
# back up once the pending permission is resolved.
_pending_calls: dict[str, asyncio.Task] = {}


def _get_permission_queue(session_id: str) -> asyncio.Queue:
    queue = _permission_queues.get(session_id)
    if queue is None:
        queue = asyncio.Queue()
        _permission_queues[session_id] = queue
    return queue


async def _wait_for_permission_event(session_id: str) -> dict[str, Any]:
    return await _get_permission_queue(session_id).get()


async def _watch_permission_events(session_id: str, base_url: str) -> None:
    """
    Long-lived per-session task: subscribes to opencode's own SSE event
    stream and forwards every permission.asked event for this session onto
    its queue, so a pending prompt can be raced against it instead of
    blocking silently until OPENCODE_PROMPT_TIMEOUT_SECONDS.
    """
    queue = _get_permission_queue(session_id)
    try:
        async with httpx2.AsyncClient(timeout=None) as client:
            async with client.stream("GET", f"{base_url}/event") as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[len("data: "):])
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") != "permission.asked":
                        continue
                    properties = event.get("properties")
                    if isinstance(properties, dict) and properties.get("sessionID") == session_id:
                        await queue.put(properties)
    except Exception:
        logger.exception("Permission event watcher died for session %s", session_id)


# Keeps references to fire-and-forget background tasks (stdout drains) alive
# for as long as the process runs -- otherwise they could be garbage
# collected mid-flight.
_background_tasks: set[asyncio.Task] = set()


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


# Connection-level failures (refused, reset, DNS, or the initial connect
# itself timing out) mean the server is actually gone/unreachable and worth
# restarting. Deliberately excludes ReadTimeout/WriteTimeout/PoolTimeout: a
# long-running but still-connected prompt (up to timeout_seconds) must not
# be torn down and retried just because it's slow.
_TRANSIENT_SERVER_ERRORS = (httpx2.NetworkError, httpx2.ConnectTimeout)


def _evict_and_kill_session_backend(session_id: str) -> None:
    """Stop tracking a session's server that failed to respond, and actually
    kill it so it doesn't leak as an untracked, unkillable orphan process."""
    cached = _session_state.pop(session_id, None)

    pending_task = _pending_calls.pop(session_id, None)
    if pending_task is not None:
        pending_task.cancel()
    _permission_queues.pop(session_id, None)

    if cached is None:
        return

    process, _base_url, _handle = cached

    if process.returncode is not None:
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        logger.exception("Failed to terminate unresponsive opencode serve process %s", process.pid)


def _kill_opencode_servers() -> None:
    """Best-effort synchronous cleanup, registered with atexit."""
    for process, _base_url, _handle in _session_state.values():
        if process.returncode is not None:
            continue
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            logger.exception("Failed to terminate opencode serve process %s", process.pid)


atexit.register(_kill_opencode_servers)


def _install_termination_cleanup() -> None:
    """
    atexit alone is not enough: an MCP host commonly tears down this server's
    process with SIGTERM (e.g. on /mcp reconnect) rather than letting the
    interpreter exit normally, and atexit handlers do not run on a signal.
    Chain our cleanup in front of whatever handler (default or the MCP SDK's
    own) would otherwise run, so opencode serve is still killed either way.
    SIGKILL cannot be handled by any process and is an accepted exception.
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handler = signal.getsignal(sig)

        def handler(signum, frame, _previous=previous_handler):
            _kill_opencode_servers()
            if callable(_previous):
                _previous(signum, frame)
            elif _previous == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            # SIG_IGN: leave the signal ignored, same as before.

        signal.signal(sig, handler)


_install_termination_cleanup()


async def _create_session_pane(agent: str, project_dir: Path) -> tuple[str, str, str] | None:
    """
    Best-effort: create an empty Orca terminal to host this session's visible
    `opencode attach` view, returning (handle, pane_key, tab_id). Returns None
    (never raises) if Orca is unavailable -- a missing pane just means no
    visible terminal and no live status, not a failed implement/review call.
    """
    try:
        returncode, stdout, stderr = await run_command(
            [
                "orca",
                "terminal",
                "create",
                "--worktree",
                "current",
                "--title",
                f"OpenCode {agent}",
                "--json",
            ],
            cwd=str(project_dir),
            timeout=ORCA_COMMAND_TIMEOUT_SECONDS,
        )

        if returncode != 0:
            raise RuntimeError(stderr or stdout)

        terminal = parse_json_output(stdout).get("result")
        terminal = terminal.get("terminal") if isinstance(terminal, dict) else None
        handle = terminal.get("handle") if isinstance(terminal, dict) else None
        pane_key = terminal.get("paneKey") if isinstance(terminal, dict) else None
        tab_id = terminal.get("tabId") if isinstance(terminal, dict) else None

        if not (handle and pane_key and tab_id):
            raise RuntimeError(f"Orca did not return full pane identity:\n{stdout}")

        return str(handle), str(pane_key), str(tab_id)
    except Exception:
        logger.exception("Failed to create a visibility pane for agent %s", agent)
        return None


def _session_server_env(pane: tuple[str, str, str] | None) -> dict[str, str]:
    """
    Env for a session's dedicated opencode serve process. The pane-identity
    vars are always removed first. If `pane` (handle, pane_key, tab_id) is
    given, they're re-set to point at that terminal, so opencode's Orca
    status-hook plugin (loaded via ORCA_OPENCODE_CONFIG_DIR, left untouched)
    attributes busy/idle/waiting status to it. Otherwise they stay stripped so
    nothing is misattributed to whatever pane this MCP process itself
    inherited (e.g. its own coordinator terminal).
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in ("ORCA_PANE_KEY", "ORCA_TAB_ID", "ORCA_TERMINAL_HANDLE", "ORCA_AGENT_LAUNCH_TOKEN")
    }

    if pane is not None:
        handle, pane_key, tab_id = pane
        env["ORCA_PANE_KEY"] = pane_key
        env["ORCA_TAB_ID"] = tab_id
        env["ORCA_TERMINAL_HANDLE"] = handle

    return env


async def _spawn_session_server(
    project_dir: Path, pane: tuple[str, str, str] | None
) -> tuple[asyncio.subprocess.Process, str]:
    process = await asyncio.create_subprocess_exec(
        "opencode",
        "serve",
        "--port",
        "0",
        "--hostname",
        "127.0.0.1",
        cwd=str(project_dir),
        env=_session_server_env(pane),
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
                # EOF on the pipe can arrive slightly before asyncio updates
                # returncode; wait for it so the message reports the real
                # exit code instead of a misleading "None".
                await process.wait()
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


async def _attach_terminal_to_session(handle: str, base_url: str, session_id: str) -> None:
    """Best-effort: point the pane at the session now that both exist. Never
    raises -- a failure here must never fail the actual implement/review call."""
    try:
        returncode, stdout, stderr = await run_command(
            [
                "orca",
                "terminal",
                "send",
                "--terminal",
                handle,
                "--text",
                f"opencode attach {base_url} --session {session_id}",
                "--enter",
                "--json",
            ],
            timeout=ORCA_COMMAND_TIMEOUT_SECONDS,
        )

        if returncode != 0:
            raise RuntimeError(stderr or stdout)
    except Exception:
        logger.exception("Failed to attach terminal %s to session %s", handle, session_id)


async def _create_session_backend(
    session_id: str | None, agent: str, project_dir: Path
) -> tuple[str, str]:
    pane = await _create_session_pane(agent, project_dir)
    process, base_url = await _spawn_session_server(project_dir, pane)

    if session_id is None:
        session_id = await create_opencode_session(base_url, title=f"orca-bridge {agent}")

    handle = pane[0] if pane is not None else None
    _session_state[session_id] = (process, base_url, handle)
    _spawn_background_task(_watch_permission_events(session_id, base_url))

    if handle is not None:
        await _attach_terminal_to_session(handle, base_url, session_id)

    return base_url, session_id


async def ensure_session_backend(
    session_id: str | None, agent: str, project_dir: Path
) -> tuple[str, str]:
    """Reuse this session's dedicated server/terminal if both are still
    alive; otherwise (re)create them, preserving the session_id."""
    if session_id is None:
        return await _create_session_backend(None, agent, project_dir)

    async with _get_session_lock(session_id):
        cached = _session_state.get(session_id)

        if cached is not None:
            process, base_url, handle = cached
            server_alive = process.returncode is None
            terminal_alive = True

            if server_alive and handle is not None:
                returncode, _, _ = await run_command(
                    ["orca", "terminal", "show", "--terminal", handle, "--json"],
                    timeout=ORCA_COMMAND_TIMEOUT_SECONDS,
                )
                terminal_alive = returncode == 0

            if server_alive and terminal_alive:
                return base_url, session_id

        return await _create_session_backend(session_id, agent, project_dir)


async def create_opencode_session(base_url: str, title: str) -> str:
    async with httpx2.AsyncClient(timeout=ORCA_COMMAND_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{base_url}/session", json={"title": title})

    if response.is_error:
        raise RuntimeError(
            f"Failed to create OpenCode session:\n{response.text}"
        )

    data = response.json()

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unexpected OpenCode response (not an object):\n{response.text}"
        )

    session_id = data.get("id")

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

    data = response.json()

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unexpected OpenCode response (not an object):\n{response.text}"
        )

    return data


def _finalize_prompt_result(session_id: str, result: dict[str, Any]) -> dict[str, Any]:
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


def _permission_required_result(session_id: str, request: dict[str, Any]) -> dict[str, Any]:
    request_id = request.get("id")
    action = request.get("permission")
    patterns = request.get("patterns")
    return {
        "session_id": session_id,
        "status": "permission_required",
        "request_id": request_id,
        "action": action,
        "patterns": patterns,
        "message": (
            f"OpenCode is asking permission to {action} ({patterns}) and nobody is "
            "watching interactively. Decide, then call answer_permission("
            f"session_id={session_id!r}, request_id={request_id!r}, "
            "reply='once'|'always'|'reject') to resolve it."
        ),
    }


async def run_opencode(
    agent: str,
    prompt: str,
    session_id: str | None = None,
    timeout_seconds: int = OPENCODE_PROMPT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    project_dir = get_project_dir()
    current_session_id = session_id

    # Two attempts: if the session's backend turns out to be unreachable
    # (alive per its exit code, but hung or otherwise not answering),
    # invalidate it and retry once against a freshly spawned one before
    # giving up. The (possibly newly created) session_id is preserved across
    # the retry -- opencode session data persists independently of which
    # server process created it, so conversation history isn't lost.
    for attempt in range(2):
        try:
            base_url, current_session_id = await ensure_session_backend(
                current_session_id, agent, project_dir
            )

            # Race the prompt against opencode's own permission-event stream
            # instead of just awaiting it: nobody is watching this session
            # interactively, so a pending "ask" must be surfaced back to the
            # caller instead of sitting until timeout_seconds elapses.
            send_task = asyncio.ensure_future(
                send_opencode_prompt(base_url, current_session_id, agent, prompt, timeout_seconds)
            )
            permission_task = asyncio.ensure_future(
                _wait_for_permission_event(current_session_id)
            )
            done, _pending = await asyncio.wait(
                {send_task, permission_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if send_task not in done:
                _pending_calls[current_session_id] = send_task
                return _permission_required_result(current_session_id, permission_task.result())

            permission_task.cancel()
            result = await send_task
        except _TRANSIENT_SERVER_ERRORS:
            if attempt == 1:
                raise
            logger.exception(
                "Lost connection to opencode serve for session %s; restarting and retrying once",
                current_session_id,
            )
            if current_session_id is not None:
                _evict_and_kill_session_backend(current_session_id)
            continue

        session_id = current_session_id
        break

    return _finalize_prompt_result(session_id, result)


async def _answer_permission(
    session_id: str, request_id: str, reply: str, message: str | None
) -> dict[str, Any]:
    if reply not in ("once", "always", "reject"):
        raise ToolError(f"Invalid reply {reply!r}; expected 'once', 'always', or 'reject'")

    cached = _session_state.get(session_id)
    if cached is None:
        raise ToolError(f"Unknown or expired session_id: {session_id}")
    _, base_url, _ = cached

    body: dict[str, Any] = {"response": reply}
    if message:
        body["message"] = message

    async with httpx2.AsyncClient(timeout=ORCA_COMMAND_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{base_url}/session/{session_id}/permissions/{request_id}",
            json=body,
        )

    if response.is_error:
        raise RuntimeError(
            f"Failed to answer permission (status {response.status_code}):\n{response.text}"
        )

    send_task = _pending_calls.get(session_id)
    if send_task is None:
        return {"session_id": session_id, "status": "ok"}

    # The prompt this permission was blocking may hit another "ask" before
    # it finishes -- race it the same way run_opencode() does, so a chain of
    # permission requests stays entirely between Claude and this tool.
    permission_task = asyncio.ensure_future(_wait_for_permission_event(session_id))
    done, _pending = await asyncio.wait(
        {send_task, permission_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if send_task not in done:
        return _permission_required_result(session_id, permission_task.result())

    permission_task.cancel()
    del _pending_calls[session_id]
    return _finalize_prompt_result(session_id, await send_task)


@mcp.tool()
async def answer_permission(
    session_id: str, request_id: str, reply: str, message: str | None = None
) -> dict[str, Any]:
    """
    Resolve a pending OpenCode permission request (returned as
    status="permission_required" by any OpenCode agent tool)
    instead of leaving the underlying opencode session blocked waiting for
    an interactive answer nobody will give it.

    reply is "once" (allow this one time), "always" (allow and remember for
    the rest of this session), or "reject" (deny). Returns the agent's final
    result once resolved, or another status="permission_required" if the
    resumed call hits a second pending permission.
    """
    try:
        return await _answer_permission(session_id, request_id, reply, message)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("answer_permission() failed")
        raise ToolError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Dynamic agent discovery.
#
# Each OpenCode agent (~/.config/opencode/agent/*.md, overridden per-project
# by <project_dir>/.opencode/agent/*.md) becomes its own MCP tool, named and
# described from the agent's own frontmatter. Adding an agent is then just
# dropping a .md file -- no server.py change needed.
# ---------------------------------------------------------------------------

AGENT_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def _parse_agent_frontmatter(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning("Skipping agent %s: could not read file", path)
        return None

    match = AGENT_FRONTMATTER_RE.match(text)
    if not match:
        logger.warning("Skipping agent %s: no valid frontmatter delimiter", path)
        return None
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        logger.warning("Skipping agent %s: invalid YAML frontmatter", path)
        return None

    if not isinstance(data, dict):
        logger.warning("Skipping agent %s: frontmatter is not a YAML object", path)
        return None

    return data


def discover_agents(project_dir: Path) -> dict[str, str]:
    """
    Scan the global and project-local OpenCode agent directories and return
    {agent_slug: description}. Project-local agents (<project_dir>/.opencode/agent)
    override global ones (~/.config/opencode/agent) of the same name.
    """
    agents: dict[str, str] = {}

    for directory in (
        Path.home() / ".config" / "opencode" / "agent",
        project_dir / ".opencode" / "agent",
    ):
        if not directory.is_dir():
            continue
        for md_file in sorted(directory.glob("*.md")):
            frontmatter = _parse_agent_frontmatter(md_file)
            if frontmatter is None:
                continue
            description = frontmatter.get("description")
            agents[md_file.stem] = (
                str(description) if description else f"Run the '{md_file.stem}' OpenCode agent"
            )

    return agents


def _register_agent_tool(name: str, description: str) -> None:
    async def handler(plan: str, session_id: str | None = None) -> dict[str, Any]:
        try:
            return await run_opencode(agent=name, session_id=session_id, prompt=plan)
        except Exception as exc:
            logger.exception("%s() failed", name)
            raise ToolError(str(exc)) from exc

    handler.__name__ = name
    mcp.add_tool(handler, name=name, description=description)


for _agent_name, _agent_description in discover_agents(get_project_dir()).items():
    _register_agent_tool(_agent_name, _agent_description)


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
