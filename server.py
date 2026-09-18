import asyncio
import json
import logging
import os
import re
import shlex
import signal
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

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


def make_exit_marker_prefix(token: str) -> str:
    return f"ORCA_BRIDGE_EXIT_{token}:"


def extract_exit_marker_code(line: str, marker_prefix: str) -> str | None:
    match = re.search(re.escape(marker_prefix) + r"(-?\d+)", line)
    return match.group(1) if match else None


def build_opencode_command(
    agent: str,
    prompt: str,
    exit_marker_token: str,
    session_id: str | None = None,
) -> str:
    """
    Build the command executed inside the Orca terminal.

    OpenCode remains responsible for its own agent/model configuration.
    We intentionally do not pass --model.
    """

    command = [
        "opencode",
        "run",
        "--agent",
        agent,
        "--format",
        "json",
    ]

    if session_id:
        command.extend([
            "--session",
            session_id,
        ])

    command.append(prompt)

    # OpenCode can exit (crash, or give up after a rejected tool call)
    # without ever emitting a "step_finish"/stop JSON event, and Orca's
    # terminal.status stays "running" even after the underlying command
    # has returned to the shell (the terminal itself, i.e. the shell, is
    # still alive). This marker is the only reliable way to detect that
    # the opencode process itself has terminated, clean or not.
    #
    # The token is per-run and random (not a fixed string) because a fixed
    # marker can appear inside a reviewed file's own content -- e.g. this
    # very source file -- and get echoed back through a "read" tool event,
    # producing a false match.
    marker_prefix = make_exit_marker_prefix(exit_marker_token)

    return (
        f"{shlex.join(command)}; "
        f"printf '\\n{marker_prefix}%s\\n' \"$?\""
    )


def parse_opencode_event(line: str) -> dict[str, Any] | None:
    """
    Orca returns OpenCode JSONL events as strings inside terminal.tail.
    """
    try:
        value = json.loads(line)
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(value, dict):
        return None

    return value


def extract_event_text(event: dict[str, Any]) -> str | None:
    if event.get("type") != "text":
        return None

    part = event.get("part")

    if not isinstance(part, dict):
        return None

    text = part.get("text")

    if not text:
        return None

    return str(text)


# step_finish reasons that mean the run is actually over. "tool-calls" (or
# similar) means the model is about to continue after a tool result -- more
# steps follow, so it must NOT be treated as completion.
TERMINAL_STEP_FINISH_REASONS = {"stop", "length", "content_filter", "content-filter"}


def get_step_finish_reason(event: dict[str, Any]) -> str | None:
    """
    "stop" and truncation-style reasons ("length", "content_filter") end
    the run, so the caller gets the partial response instead of the loop
    spinning until the 900s timeout with a misleading "Timed out" error.
    Non-terminal reasons (e.g. "tool-calls") return None so the loop keeps
    waiting for the steps that follow.
    """
    if event.get("type") != "step_finish":
        return None

    part = event.get("part")

    if not isinstance(part, dict):
        return None

    reason = part.get("reason")

    if reason not in TERMINAL_STEP_FINISH_REASONS:
        return None

    return str(reason)


def extract_event_error(event: dict[str, Any]) -> str | None:
    if event.get("type") != "error":
        return None

    error = event.get("error")

    if isinstance(error, str):
        return error

    if isinstance(error, dict):
        return json.dumps(error, indent=2)

    return json.dumps(event, indent=2)


async def read_orca_terminal(
    terminal_handle: str,
) -> dict[str, Any]:
    returncode, stdout, stderr = await run_command(
        [
            "orca",
            "terminal",
            "read",
            "--terminal",
            terminal_handle,
            "--json",
        ],
        timeout=ORCA_COMMAND_TIMEOUT_SECONDS,
    )

    if returncode != 0:
        raise RuntimeError(
            "Orca terminal read failed:\n"
            f"{stderr or stdout}"
        )

    return parse_json_output(stdout)


async def close_orca_terminal(terminal_handle: str) -> None:
    """Best-effort cleanup: never let a close failure mask the real error."""
    try:
        await run_command(
            ["orca", "terminal", "close", "--terminal", terminal_handle, "--json"],
            timeout=ORCA_COMMAND_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Failed to close Orca terminal %s", terminal_handle)


async def run_opencode(
    agent: str,
    prompt: str,
    session_id: str | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """
    Run OpenCode through Orca.

    Architecture:

        MCP
          ↓
        Orca terminal
          ↓
        OpenCode batch
          ↓
        Orca terminal stream
          ↓
        MCP

    OpenCode remains non-interactive.
    The model is selected by the OpenCode agent definition.
    """

    cwd = str(get_project_dir())

    exit_marker_token = uuid.uuid4().hex
    exit_marker_prefix = make_exit_marker_prefix(exit_marker_token)

    opencode_command = build_opencode_command(
        agent=agent,
        prompt=prompt,
        exit_marker_token=exit_marker_token,
        session_id=session_id,
    )

    # Create a visible Orca terminal running OpenCode.
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
            opencode_command,
            "--json",
        ],
        cwd=cwd,
        timeout=ORCA_COMMAND_TIMEOUT_SECONDS,
    )

    if returncode != 0:
        raise RuntimeError(
            "Failed to create Orca terminal:\n"
            f"{stderr or stdout}"
        )

    create_result = parse_json_output(stdout)
    terminal_handle = extract_orca_terminal_handle(create_result)

    text_parts: list[str] = []
    detected_session_id = session_id
    # Dedup per raw line rather than per poll snapshot: whether Orca's
    # "tail" is a growing buffer or a sliding window, a line already seen
    # must not be processed twice (it would otherwise duplicate streamed
    # text, or re-raise an error/finish event that was already handled).
    processed_lines: set[str] = set()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    poll_interval = 1.0
    # Kept low (rather than backing off further) because if Orca's "tail"
    # is a bounded sliding window rather than a growing buffer, a longer
    # gap between polls risks scrolling lines out of the window unseen.
    max_poll_interval = 2.0

    while loop.time() < deadline:
        try:
            terminal_result = await read_orca_terminal(
                terminal_handle
            )

            inner_result = terminal_result.get("result")
            terminal = (
                inner_result.get("terminal") if isinstance(inner_result, dict) else None
            )
            tail = terminal.get("tail") if isinstance(terminal, dict) else None

            if not isinstance(tail, list):
                tail = []
        except Exception:
            # An Orca-level failure here (CLI error, its own
            # ORCA_COMMAND_TIMEOUT_SECONDS timeout, or an unexpectedly
            # shaped payload) leaves the opencode process's state unknown
            # -- close defensively rather than leaking a possibly
            # still-running terminal. Best-effort: this never masks the
            # original exception being re-raised below.
            await close_orca_terminal(terminal_handle)
            raise

        for raw_line in tail:
            if not isinstance(raw_line, str) or raw_line in processed_lines:
                continue

            processed_lines.add(raw_line)

            exit_code = extract_exit_marker_code(raw_line, exit_marker_prefix)

            if exit_code is not None:
                # A zero exit is a success signal from the shell itself,
                # more authoritative than the JSONL stream: the stream can
                # legitimately miss the final "stop" event if Orca's tail
                # is a bounded window that scrolled past it before a poll
                # observed it. Only a non-zero exit is treated as a real
                # failure worth raising on.
                if exit_code == "0":
                    return {
                        "session_id": detected_session_id,
                        "response": "".join(text_parts),
                        "stderr": "",
                        "terminal_handle": terminal_handle,
                        "finish_reason": "process_exit",
                    }

                # The process has already exited (that's what the marker
                # means), so this close is cosmetic tidiness rather than
                # stopping a still-running process, but it keeps cleanup
                # behavior uniform across every non-success return path.
                await close_orca_terminal(terminal_handle)

                raise RuntimeError(
                    "OpenCode exited with a non-zero status without "
                    f"producing a completion event (exit_code={exit_code}).\n"
                    f"terminal={terminal_handle}\n"
                    f"partial_response={''.join(text_parts)}"
                )

            event = parse_opencode_event(raw_line)

            if event is None:
                continue

            event_session_id = event.get("sessionID")

            if event_session_id:
                detected_session_id = str(event_session_id)

            error = extract_event_error(event)

            if error:
                # An error event does NOT imply the opencode process has
                # exited (only the exit marker is reliable evidence of
                # that) -- close defensively rather than leak a possibly
                # still-running process, same principle as the read-failure
                # and timeout paths above.
                await close_orca_terminal(terminal_handle)

                raise RuntimeError(
                    "OpenCode returned an error:\n"
                    f"{error}\n"
                    f"terminal={terminal_handle}"
                )

            text = extract_event_text(event)

            if text:
                text_parts.append(text)

            finish_reason = get_step_finish_reason(event)

            if finish_reason is not None:
                return {
                    "session_id": detected_session_id,
                    "response": "".join(text_parts),
                    "stderr": "",
                    "terminal_handle": terminal_handle,
                    "finish_reason": finish_reason,
                }

        await asyncio.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, max_poll_interval)

    # Do not use terminal.status == exited as the completion signal.
    # Orca/OpenCode currently leaves the terminal reported as running even
    # after OpenCode has returned to the shell.
    #
    # A timed-out run means the underlying process is genuinely still
    # running (unlike the success/error paths, where it has already
    # exited) -- close the terminal so it doesn't keep running/accumulating
    # forever after the bridge gives up on it.
    await close_orca_terminal(terminal_handle)

    raise RuntimeError(
        "Timed out waiting for OpenCode to finish through Orca.\n"
        f"terminal={terminal_handle}\n"
        f"session_id={detected_session_id}\n"
        f"partial_response={''.join(text_parts)}"
    )


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