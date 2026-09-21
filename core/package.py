"""
Generic OpenCode sandboxing + session management trunk.

This module knows nothing about any particular front-end integration (Orca
or otherwise): it spawns `opencode serve` inside a dedicated `sbx` (Docker
Sandboxes) sandbox per session, speaks its HTTP/SSE API, and tracks the
session <-> sandbox binding. Anything an integration needs to plug in --
giving the user a way to see the session live, or reporting busy/idle/waiting
status somewhere -- goes through the VisibilityHook/StatusHook interfaces
below, implemented per integration (see server/orca.py for the Orca one,
server/generic.py for the no-op one).
"""

import asyncio
import atexit
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import httpx2
import yaml
from mcp.server.mcpserver.exceptions import ToolError


LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "orca_bridge.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("orca_bridge")
logger.setLevel(logging.INFO)
logger.propagate = False

_log_handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(_log_handler)


def get_project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())).resolve()


COMMAND_TIMEOUT_SECONDS = 30


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
    Parse a JSON object out of a CLI command's output.

    The whole output is tried as a single JSON document first (the common,
    fast case). If that fails -- e.g. a banner/log line before or after the
    JSON -- fall back to scanning for the first "{" that starts a fully
    valid JSON document, which handles single-line JSON, pretty-printed
    multi-line JSON, and surrounding junk uniformly.
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
# Integration hooks.
#
# A session's lifecycle is entirely generic (spawn a sandbox, run
# `opencode serve` in it, talk HTTP/SSE to it), except for two things an
# integration may want to plug in:
#
#   - VisibilityHook: give the user a way to see/attach to the session (e.g.
#     Orca opens a terminal pane running `opencode attach`), and forward any
#     extra environment variables that visibility mechanism needs inside the
#     sandbox.
#   - StatusHook: report busy/waiting/idle transitions somewhere (e.g. Orca
#     posts them to its own hook HTTP server so a UI can reflect them).
#
# Both interfaces already behave as a working no-op integration -- a generic,
# non-Orca MCP server can use them completely unmodified.
# ---------------------------------------------------------------------------


class VisibilityHook:
    async def before_spawn(self, agent: str, project_dir: Path) -> Any | None:
        """Called before a session's sandbox is created. May return an opaque
        handle that after_spawn/is_alive/extra_env will receive back."""
        return None

    async def after_spawn(self, handle: Any, base_url: str, session_id: str) -> None:
        """Called once the sandbox and session_id both exist."""

    async def is_alive(self, handle: Any) -> bool:
        """Whether the visibility mechanism (e.g. a terminal pane) behind
        this handle is still alive. A dead sandbox always forces a respawn
        regardless of this return value."""
        return True

    def extra_env(self, handle: Any) -> dict[str, str]:
        """Extra environment variables to forward into the sandboxed
        `opencode serve` process."""
        return {}


class StatusHook:
    async def post_status(
        self, session_id: str, event_name: str, properties: dict[str, Any] | None = None
    ) -> None:
        """Report a busy/waiting/idle status transition for this session."""


# ---------------------------------------------------------------------------
# OpenCode server management.
#
# Architecture:
#
#     MCP  --HTTP/JSON-->  opencode serve  (data channel, drives the run),
#                          running inside a dedicated `sbx` (Docker Sandboxes)
#                          sandbox rather than as a native host process
#
# The MCP never scrapes terminal output for opencode's own responses:
# `opencode serve` gives structured JSON directly. Its HTTP port is published
# to the host by `sbx`, so `base_url` is a plain http://127.0.0.1:<port> from
# the MCP's point of view regardless of where opencode actually runs.
# ---------------------------------------------------------------------------

OPENCODE_SERVE_STARTUP_TIMEOUT_SECONDS = 15
OPENCODE_PROMPT_TIMEOUT_SECONDS = 900
SBX_COMMAND_TIMEOUT_SECONDS = 60

# Fixed port opencode serve binds to *inside* the sandbox. Each sandbox is
# its own network namespace, so this never collides across sessions -- only
# the host-side published port (resolved via `sbx ls --json`) needs to be
# unique, and `sbx` picks that one itself.
SBX_OPENCODE_CONTAINER_PORT = 4096

_LISTENING_URL_RE = re.compile(r"listening on (http://\S+)")

# session_id -> (sandbox_name, base_url, visibility_handle_or_None). Each
# session gets its own dedicated sbx sandbox running opencode serve so one
# session's sandbox can never affect another's filesystem or network
# isolation, and so an integration's visibility mechanism (if any) can be
# attributed to a single session instead of a shared server with no owner.
_session_state: dict[str, tuple[str, str, Any]] = {}

# session_id -> lock, so two concurrent calls sharing a session_id don't each
# decide "no backend yet" and spawn a duplicate server/pane. Lazily
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


async def _sbx_remove_sandbox(sandbox_name: str) -> None:
    """Best-effort: tear down a session's sandbox and everything running
    inside it (opencode serve). Never raises."""
    try:
        returncode, stdout, stderr = await run_command(
            ["sbx", "rm", "--force", sandbox_name], timeout=SBX_COMMAND_TIMEOUT_SECONDS
        )
        if returncode != 0:
            logger.warning("sbx rm failed for sandbox %s: %s", sandbox_name, stderr or stdout)
    except Exception:
        logger.exception("Failed to remove sandbox %s", sandbox_name)


async def _evict_and_kill_session_backend(session_id: str) -> None:
    """Stop tracking a session's sandbox that failed to respond, and actually
    remove it so it doesn't leak as an untracked, unkillable orphan container."""
    cached = _session_state.pop(session_id, None)

    pending_task = _pending_calls.pop(session_id, None)
    if pending_task is not None:
        pending_task.cancel()
    _permission_queues.pop(session_id, None)

    if cached is None:
        return

    sandbox_name, _base_url, _handle = cached
    await _sbx_remove_sandbox(sandbox_name)


def _kill_all_sandboxes() -> None:
    """Best-effort synchronous cleanup, registered with atexit. Runs outside
    the asyncio loop (atexit / a signal handler), so `sbx rm` is invoked as a
    plain blocking subprocess rather than through run_command."""
    for sandbox_name, _base_url, _handle in _session_state.values():
        try:
            subprocess.run(
                ["sbx", "rm", "--force", sandbox_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=SBX_COMMAND_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("Failed to remove sandbox %s", sandbox_name)


atexit.register(_kill_all_sandboxes)


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
            _kill_all_sandboxes()
            if callable(_previous):
                _previous(signum, frame)
            elif _previous == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            # SIG_IGN: leave the signal ignored, same as before.

        signal.signal(sig, handler)


_install_termination_cleanup()


def _sbx_sandbox_name() -> str:
    return f"opencode-bridge-{uuid.uuid4().hex[:12]}"


def _find_agent_definition(agent: str, project_dir: Path) -> Path | None:
    """Same precedence as discover_agents(): a project-local definition
    overrides the global one of the same name."""
    project_local = project_dir / ".opencode" / "agent" / f"{agent}.md"
    if project_local.is_file():
        return project_local
    global_file = Path.home() / ".config" / "opencode" / "agent" / f"{agent}.md"
    if global_file.is_file():
        return global_file
    return None


def _get_agent_network_profile(agent: str, project_dir: Path) -> str:
    """
    Reads the `network:` frontmatter key (see .opencode/agent/*.md) an agent
    declares for its sandbox: "balanced" (default; inherits whatever global
    sbx network policy is already configured on this machine) or "none"
    (deny all outbound traffic for this sandbox specifically).
    """
    path = _find_agent_definition(agent, project_dir)
    if path is None:
        return "balanced"
    frontmatter = _parse_agent_frontmatter(path)
    if frontmatter is None:
        return "balanced"
    profile = frontmatter.get("network")
    return profile if profile in ("balanced", "none") else "balanced"


async def _inject_agent_definition(sandbox_name: str, agent: str, source_path: Path) -> None:
    """
    A sandboxed opencode only ever sees <project_dir>/.opencode/agent (it is
    bind-mounted into the sandbox at the same path) -- it cannot resolve the
    host's global ~/.config/opencode/agent, whose location depends on a
    $HOME that doesn't exist inside the sandbox's own filesystem. If this
    agent is only defined globally, write its definition straight into the
    sandbox's own filesystem (never onto the host project) via `sbx exec`,
    piping the content over stdin -- no temp file, no project-local copy.
    Best effort: logs and continues on failure rather than blocking sandbox
    creation; a missing agent definition surfaces as a clear error from
    opencode itself on the next prompt instead.
    """
    try:
        content = source_path.read_text(encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            "sbx", "exec", "-i", sandbox_name, "sh", "-c",
            f'mkdir -p "$HOME/.config/opencode/agent" && '
            f'cat > "$HOME/.config/opencode/agent/{agent}.md"',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate(input=content.encode("utf-8"))
        if process.returncode != 0:
            logger.warning(
                "Failed to inject agent definition for %s into sandbox %s: %s",
                agent, sandbox_name, stderr.decode(errors="replace"),
            )
    except OSError:
        logger.exception("Failed to inject agent definition for %s into sandbox %s", agent, sandbox_name)


def _sandbox_exec_env(handle: Any, visibility_hook: VisibilityHook) -> dict[str, str]:
    """
    Explicit env vars forwarded into the sandbox for `opencode serve`. Unlike
    the native-process world this replaces, an isolated `sbx exec` doesn't
    need the whole host environment preserved-and-scrubbed -- only the small
    set of OPENCODE_CONFIG_DIR plus whatever the visibility hook itself needs
    forwarded (e.g. Orca's own pane identity).
    """
    env: dict[str, str] = {}

    hooks_dir = os.environ.get("OPENCODE_CONFIG_DIR")
    if hooks_dir:
        env["OPENCODE_CONFIG_DIR"] = hooks_dir

    env.update(visibility_hook.extra_env(handle))

    return env


async def _sbx_inspect(sandbox_name: str) -> dict[str, Any] | None:
    """Returns this sandbox's `sbx ls --json` entry, or None if it doesn't
    exist (removed, never created, or sandboxd doesn't know about it)."""
    returncode, stdout, stderr = await run_command(
        ["sbx", "ls", "--json"], timeout=SBX_COMMAND_TIMEOUT_SECONDS
    )
    if returncode != 0:
        raise RuntimeError(f"sbx ls failed:\n{stderr or stdout}")

    data = parse_json_output(stdout)
    for sandbox in data.get("sandboxes", []) or []:
        if isinstance(sandbox, dict) and sandbox.get("name") == sandbox_name:
            return sandbox
    return None


async def _sbx_published_port(sandbox_name: str, container_port: int) -> int | None:
    sandbox = await _sbx_inspect(sandbox_name)
    if sandbox is None:
        return None
    for port in sandbox.get("ports", []) or []:
        if isinstance(port, dict) and port.get("sandbox_port") == container_port:
            return port.get("host_port")
    return None


async def _spawn_session_server(
    project_dir: Path, handle: Any, agent: str, visibility_hook: VisibilityHook
) -> tuple[asyncio.subprocess.Process, str, str]:
    """
    Creates a dedicated sbx sandbox for this session and starts
    `opencode serve` inside it. Returns (exec_process, base_url,
    sandbox_name).

    `exec_process` wraps the `sbx exec -d ... opencode serve` invocation
    itself, kept only so its stdout can be drained for the process lifetime
    (see _drain_stream below) -- it is NOT used to determine liveness.
    Unlike a native subprocess, `sbx exec -d` stays attached and streams the
    child's output for as long as it runs rather than detaching the CLI
    itself, so ensure_session_backend asks sbx directly (`sbx ls --json`)
    instead of checking this process's returncode.
    """
    sandbox_name = _sbx_sandbox_name()
    agent_definition = _find_agent_definition(agent, project_dir)
    network_profile = _get_agent_network_profile(agent, project_dir)

    create_args = ["sbx", "create", "opencode", str(project_dir)]
    hooks_dir = os.environ.get("OPENCODE_CONFIG_DIR")
    if hooks_dir:
        create_args.append(f"{hooks_dir}:ro")
    create_args += [
        "--name", sandbox_name,
        "--publish", str(SBX_OPENCODE_CONTAINER_PORT),
        "--quiet",
    ]
    if network_profile == "none":
        create_args += ["--deny-network", "**"]

    returncode, stdout, stderr = await run_command(create_args, timeout=SBX_COMMAND_TIMEOUT_SECONDS)
    if returncode != 0:
        raise RuntimeError(f"sbx create failed for sandbox {sandbox_name}:\n{stderr or stdout}")

    project_local = project_dir / ".opencode" / "agent" / f"{agent}.md"
    if agent_definition is not None and agent_definition != project_local:
        await _inject_agent_definition(sandbox_name, agent, agent_definition)

    exec_env = _sandbox_exec_env(handle, visibility_hook)
    exec_args = ["sbx", "exec", "-d"]
    for key, value in exec_env.items():
        exec_args += ["-e", f"{key}={value}"]
    exec_args += [
        sandbox_name, "opencode", "serve",
        "--hostname", "0.0.0.0",
        "--port", str(SBX_OPENCODE_CONTAINER_PORT),
    ]

    process = await asyncio.create_subprocess_exec(
        *exec_args,
        stdout=asyncio.subprocess.PIPE,
        # Merged into stdout: we don't know in advance which stream carries
        # the startup banner, and only one stream needs draining afterwards.
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )

    async def read_until_listening() -> None:
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

            if _LISTENING_URL_RE.search(line.decode(errors="replace")):
                return

    try:
        await asyncio.wait_for(
            read_until_listening(), timeout=OPENCODE_SERVE_STARTUP_TIMEOUT_SECONDS
        )
        host_port = await _sbx_published_port(sandbox_name, SBX_OPENCODE_CONTAINER_PORT)
        if host_port is None:
            raise RuntimeError(f"sbx did not publish a host port for sandbox {sandbox_name}")
    except (asyncio.TimeoutError, RuntimeError):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        await _sbx_remove_sandbox(sandbox_name)
        raise

    base_url = f"http://127.0.0.1:{host_port}"

    _spawn_background_task(_drain_stream(process.stdout))

    return process, base_url, sandbox_name


async def _create_session_backend(
    session_id: str | None,
    agent: str,
    project_dir: Path,
    visibility_hook: VisibilityHook,
) -> tuple[str, str]:
    handle = await visibility_hook.before_spawn(agent, project_dir)
    _process, base_url, sandbox_name = await _spawn_session_server(
        project_dir, handle, agent, visibility_hook
    )

    if session_id is None:
        session_id = await create_opencode_session(base_url, title=f"opencode-bridge {agent}")

    _session_state[session_id] = (sandbox_name, base_url, handle)
    _spawn_background_task(_watch_permission_events(session_id, base_url))

    await visibility_hook.after_spawn(handle, base_url, session_id)

    return base_url, session_id


async def ensure_session_backend(
    session_id: str | None,
    agent: str,
    project_dir: Path,
    visibility_hook: VisibilityHook,
) -> tuple[str, str]:
    """Reuse this session's dedicated sandbox/visibility handle if both are
    still alive; otherwise (re)create them, preserving the session_id."""
    if session_id is None:
        return await _create_session_backend(None, agent, project_dir, visibility_hook)

    async with _get_session_lock(session_id):
        cached = _session_state.get(session_id)

        if cached is not None:
            sandbox_name, base_url, handle = cached
            sandbox = await _sbx_inspect(sandbox_name)
            server_alive = sandbox is not None and sandbox.get("status") == "running"
            handle_alive = True

            if server_alive:
                handle_alive = await visibility_hook.is_alive(handle)

            if server_alive and handle_alive:
                return base_url, session_id

        return await _create_session_backend(session_id, agent, project_dir, visibility_hook)


async def create_opencode_session(base_url: str, title: str) -> str:
    async with httpx2.AsyncClient(timeout=COMMAND_TIMEOUT_SECONDS) as client:
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
    visibility_hook: VisibilityHook | None = None,
    status_hook: StatusHook | None = None,
) -> dict[str, Any]:
    visibility_hook = visibility_hook or VisibilityHook()
    status_hook = status_hook or StatusHook()

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
                current_session_id, agent, project_dir, visibility_hook
            )
            await status_hook.post_status(
                current_session_id, "SessionBusy", {"sessionID": current_session_id}
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
                permission_request = permission_task.result()
                await status_hook.post_status(current_session_id, "PermissionRequest", permission_request)
                return _permission_required_result(current_session_id, permission_request)

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
                await _evict_and_kill_session_backend(current_session_id)
            continue

        session_id = current_session_id
        break

    try:
        return _finalize_prompt_result(session_id, result)
    finally:
        # The run is done (success or an application-level error from
        # OpenCode) -- tell the integration the session is idle again before
        # tearing down its sandbox, then do so now rather than leaving it
        # running until the whole MCP process exits. A future call reusing
        # this session_id will simply recreate the sandbox; OpenCode's own
        # session data lives on disk in project_dir, not in the sandbox.
        await status_hook.post_status(session_id, "SessionIdle", {"sessionID": session_id})
        await _evict_and_kill_session_backend(session_id)


async def _answer_permission(
    session_id: str,
    request_id: str,
    reply: str,
    message: str | None,
    status_hook: StatusHook,
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

    async with httpx2.AsyncClient(timeout=COMMAND_TIMEOUT_SECONDS) as client:
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

    await status_hook.post_status(session_id, "SessionBusy", {"sessionID": session_id})

    # The prompt this permission was blocking may hit another "ask" before
    # it finishes -- race it the same way run_opencode() does, so a chain of
    # permission requests stays entirely between the caller and this tool.
    permission_task = asyncio.ensure_future(_wait_for_permission_event(session_id))
    done, _pending = await asyncio.wait(
        {send_task, permission_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if send_task not in done:
        permission_request = permission_task.result()
        await status_hook.post_status(session_id, "PermissionRequest", permission_request)
        return _permission_required_result(session_id, permission_request)

    permission_task.cancel()
    del _pending_calls[session_id]
    try:
        return _finalize_prompt_result(session_id, await send_task)
    finally:
        await status_hook.post_status(session_id, "SessionIdle", {"sessionID": session_id})
        await _evict_and_kill_session_backend(session_id)


# ---------------------------------------------------------------------------
# Dynamic agent discovery.
#
# Each OpenCode agent (~/.config/opencode/agent/*.md, overridden per-project
# by <project_dir>/.opencode/agent/*.md) becomes its own MCP tool, named and
# described from the agent's own frontmatter. Adding an agent is then just
# dropping a .md file -- no code change needed.
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


ANSWER_PERMISSION_DESCRIPTION = """
Resolve a pending OpenCode permission request (returned as
status="permission_required" by any OpenCode agent tool) instead of leaving
the underlying opencode session blocked waiting for an interactive answer
nobody will give it.

reply is "once" (allow this one time), "always" (allow and remember for the
rest of this session), or "reject" (deny). Returns the agent's final result
once resolved, or another status="permission_required" if the resumed call
hits a second pending permission.
""".strip()


def _register_agent_tool(
    mcp: Any,
    name: str,
    description: str,
    visibility_hook: VisibilityHook,
    status_hook: StatusHook,
) -> None:
    async def handler(plan: str, session_id: str | None = None) -> dict[str, Any]:
        try:
            return await run_opencode(
                agent=name,
                session_id=session_id,
                prompt=plan,
                visibility_hook=visibility_hook,
                status_hook=status_hook,
            )
        except Exception as exc:
            logger.exception("%s() failed", name)
            raise ToolError(str(exc)) from exc

    handler.__name__ = name
    mcp.add_tool(handler, name=name, description=description)


def _register_answer_permission_tool(mcp: Any, status_hook: StatusHook) -> None:
    async def answer_permission(
        session_id: str, request_id: str, reply: str, message: str | None = None
    ) -> dict[str, Any]:
        try:
            return await _answer_permission(session_id, request_id, reply, message, status_hook)
        except ToolError:
            raise
        except Exception as exc:
            logger.exception("answer_permission() failed")
            raise ToolError(str(exc)) from exc

    mcp.add_tool(
        answer_permission, name="answer_permission", description=ANSWER_PERMISSION_DESCRIPTION
    )


def register_agent_tools(
    mcp: Any,
    visibility_hook: VisibilityHook | None = None,
    status_hook: StatusHook | None = None,
) -> None:
    """
    Discover every available OpenCode agent and register it as an MCP tool
    on `mcp`, plus the `answer_permission` tool they all share -- all wired
    to the given integration hooks (default: no-op, for a plain non-Orca
    MCP server).
    """
    visibility_hook = visibility_hook or VisibilityHook()
    status_hook = status_hook or StatusHook()

    project_dir = get_project_dir()
    for agent_name, agent_description in discover_agents(project_dir).items():
        _register_agent_tool(mcp, agent_name, agent_description, visibility_hook, status_hook)

    _register_answer_permission_tool(mcp, status_hook)
