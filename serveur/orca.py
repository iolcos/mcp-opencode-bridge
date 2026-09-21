"""
MCP server specialized for Orca: wires the generic OpenCode sandboxing trunk
(core/package.py) to Orca-specific visibility (a terminal pane running
`opencode attach`) and status reporting (Orca's own hook HTTP server).
"""

import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx2
from mcp.server.mcpserver import MCPServer

from core import package

mcp = MCPServer("orca-bridge")

ORCA_COMMAND_TIMEOUT_SECONDS = 30
ORCA_HOOK_POST_TIMEOUT_SECONDS = 2.0

# (terminal_handle, pane_key, tab_id), the opaque visibility handle this
# integration hands back to core/package.py.
OrcaHandle = tuple[str, str, str]


class OrcaVisibilityHook(package.VisibilityHook):
    """Gives the user a visible terminal pane (`opencode attach`) for each
    session. Best-effort throughout: a missing/dead Orca terminal just means
    no visible pane, never a failed agent call."""

    def __init__(self) -> None:
        # session_id -> (pane_key, tab_id), the identity Orca's hook HTTP
        # server uses to route a status update to the right terminal pane
        # (see post() in orca-opencode-status.js). Consumed by
        # OrcaStatusHook. Safe to grow unboundedly: each entry is a bare
        # tuple of two strings.
        self._identity: dict[str, tuple[str, str]] = {}

    async def before_spawn(self, agent: str, project_dir: Path) -> OrcaHandle | None:
        try:
            returncode, stdout, stderr = await package.run_command(
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

            terminal = package.parse_json_output(stdout).get("result")
            terminal = terminal.get("terminal") if isinstance(terminal, dict) else None
            handle = terminal.get("handle") if isinstance(terminal, dict) else None
            pane_key = terminal.get("paneKey") if isinstance(terminal, dict) else None
            tab_id = terminal.get("tabId") if isinstance(terminal, dict) else None

            if not (handle and pane_key and tab_id):
                raise RuntimeError(f"Orca did not return full pane identity:\n{stdout}")

            return str(handle), str(pane_key), str(tab_id)
        except Exception:
            package.logger.exception("Failed to create a visibility pane for agent %s", agent)
            return None

    async def after_spawn(self, handle: OrcaHandle | None, base_url: str, session_id: str) -> None:
        if handle is None:
            return

        terminal_handle, pane_key, tab_id = handle
        self._identity[session_id] = (pane_key, tab_id)

        try:
            returncode, stdout, stderr = await package.run_command(
                [
                    "orca",
                    "terminal",
                    "send",
                    "--terminal",
                    terminal_handle,
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
            package.logger.exception(
                "Failed to attach terminal %s to session %s", terminal_handle, session_id
            )

    async def is_alive(self, handle: OrcaHandle | None) -> bool:
        if handle is None:
            return True

        terminal_handle, _pane_key, _tab_id = handle
        returncode, _, _ = await package.run_command(
            ["orca", "terminal", "show", "--terminal", terminal_handle, "--json"],
            timeout=ORCA_COMMAND_TIMEOUT_SECONDS,
        )
        return returncode == 0

    def extra_env(self, handle: OrcaHandle | None) -> dict[str, str]:
        if handle is None:
            return {}

        terminal_handle, pane_key, tab_id = handle
        env = {
            "ORCA_PANE_KEY": pane_key,
            "ORCA_TAB_ID": tab_id,
            "ORCA_TERMINAL_HANDLE": terminal_handle,
        }
        for key in ("ORCA_AGENT_LAUNCH_TOKEN", "ORCA_WORKTREE_ID"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def identity_for(self, session_id: str) -> tuple[str, str] | None:
        return self._identity.get(session_id)


def _read_orca_hook_endpoint() -> tuple[str, str, str, str] | None:
    """
    Best-effort resolution of Orca's local hook-server coordinates (port,
    token, env, version), mirroring resolveHookCoords() in the
    orca-opencode-status plugin: prefer the on-disk endpoint file
    (ORCA_AGENT_HOOK_ENDPOINT), falling back to the ORCA_AGENT_HOOK_* env
    vars.
    """
    port = None
    token = None
    env_name = None
    version = None

    endpoint_path = os.environ.get("ORCA_AGENT_HOOK_ENDPOINT")
    if endpoint_path:
        try:
            for line in Path(endpoint_path).read_text(encoding="utf-8").splitlines():
                match = re.match(r"^(?:set\s+)?([A-Z0-9_]+)=(.*)$", line)
                if not match:
                    continue
                key, value = match.group(1), match.group(2).rstrip("\r")
                if key == "ORCA_AGENT_HOOK_PORT":
                    port = value
                elif key == "ORCA_AGENT_HOOK_TOKEN":
                    token = value
                elif key == "ORCA_AGENT_HOOK_ENV":
                    env_name = value
                elif key == "ORCA_AGENT_HOOK_VERSION":
                    version = value
        except OSError:
            pass

    port = port or os.environ.get("ORCA_AGENT_HOOK_PORT")
    token = token or os.environ.get("ORCA_AGENT_HOOK_TOKEN")
    env_name = env_name or os.environ.get("ORCA_AGENT_HOOK_ENV") or ""
    version = version or os.environ.get("ORCA_AGENT_HOOK_VERSION") or ""

    if not port or not token:
        return None
    return port, token, env_name, version


class OrcaStatusHook(package.StatusHook):
    """
    Posts busy/waiting/idle status updates to Orca's hook HTTP server,
    replicating post() in the orca-opencode-status plugin (see
    resolveHookCoords()/post() in orca-opencode-status.js). The plugin
    normally does this itself from inside opencode, but it cannot anymore
    now that opencode runs inside an sbx sandbox: sbx's network policy
    blocks a sandbox from ever reaching the host's own loopback (confirmed
    -- even host.docker.internal is rejected with "blocked by network
    policy: domain localhost"), so the plugin's own
    http://127.0.0.1:<hook_port> POST can never land. This MCP server runs
    on the host and has no such restriction, so it posts on the plugin's
    behalf instead, using the pane identity captured by OrcaVisibilityHook
    when this session's terminal pane was created. This only covers the
    busy/waiting/idle transitions tied to actual MCP tool-call boundaries --
    not the plugin's own finer-grained child-session and streaming
    message-preview behavior.
    """

    def __init__(self, visibility_hook: OrcaVisibilityHook) -> None:
        self._visibility_hook = visibility_hook

    async def post_status(
        self, session_id: str, event_name: str, properties: dict[str, Any] | None = None
    ) -> None:
        hook_identity = self._visibility_hook.identity_for(session_id)
        if hook_identity is None:
            return
        pane_key, tab_id = hook_identity

        coords = _read_orca_hook_endpoint()
        if coords is None:
            return
        port, token, env_name, version = coords

        body = {
            "paneKey": pane_key,
            "launchToken": os.environ.get("ORCA_AGENT_LAUNCH_TOKEN", ""),
            "tabId": tab_id,
            "worktreeId": os.environ.get("ORCA_WORKTREE_ID", ""),
            "env": env_name,
            "version": version,
            "payload": {"hook_event_name": event_name, **(properties or {})},
        }

        try:
            async with httpx2.AsyncClient(timeout=ORCA_HOOK_POST_TIMEOUT_SECONDS) as client:
                await client.post(
                    f"http://127.0.0.1:{port}/hook/opencode",
                    json=body,
                    headers={"X-Orca-Agent-Hook-Token": token},
                )
        except Exception:
            package.logger.exception(
                "Failed to post %s status hook for session %s", event_name, session_id
            )


_visibility_hook = OrcaVisibilityHook()
_status_hook = OrcaStatusHook(_visibility_hook)

package.register_agent_tools(mcp, visibility_hook=_visibility_hook, status_hook=_status_hook)


if __name__ == "__main__":
    mcp.run()
