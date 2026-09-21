import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import orca


class FakeResponse:
    def __init__(self, status_code: int = 200, json_data=None, text: str | None = None):
        self.status_code = status_code
        self._json_data = {} if json_data is None else json_data
        self.text = text if text is not None else str(self._json_data)
        self.is_error = status_code >= 400

    def json(self):
        return self._json_data


def make_mock_client(response: FakeResponse) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=response)
    return mock_client


class OrcaVisibilityHookBeforeSpawnTests(unittest.IsolatedAsyncioTestCase):
    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_parses_full_pane_identity(self, mock_run_command):
        payload = (
            '{"result": {"terminal": '
            '{"handle": "term_1", "paneKey": "tab:leaf", "tabId": "tab"}}}'
        )
        mock_run_command.return_value = (0, payload, "")

        hook = orca.OrcaVisibilityHook()
        handle = await hook.before_spawn("reviewer", Path("/tmp/p"))

        self.assertEqual(handle, ("term_1", "tab:leaf", "tab"))

    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_returns_none_on_command_failure(self, mock_run_command):
        mock_run_command.return_value = (1, "", "orca not running")

        hook = orca.OrcaVisibilityHook()
        handle = await hook.before_spawn("reviewer", Path("/tmp/p"))

        self.assertIsNone(handle)

    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_returns_none_on_incomplete_identity(self, mock_run_command):
        mock_run_command.return_value = (0, '{"result": {"terminal": {"handle": "term_1"}}}', "")

        hook = orca.OrcaVisibilityHook()
        handle = await hook.before_spawn("reviewer", Path("/tmp/p"))

        self.assertIsNone(handle)

    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_returns_none_on_exception(self, mock_run_command):
        mock_run_command.side_effect = RuntimeError("boom")

        hook = orca.OrcaVisibilityHook()
        handle = await hook.before_spawn("reviewer", Path("/tmp/p"))

        self.assertIsNone(handle)


class OrcaVisibilityHookAfterSpawnTests(unittest.IsolatedAsyncioTestCase):
    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_sends_attach_command_and_records_identity(self, mock_run_command):
        mock_run_command.return_value = (0, "{}", "")

        hook = orca.OrcaVisibilityHook()
        await hook.after_spawn(("term_1", "pane_1", "tab_1"), "http://x", "ses_1")

        args = mock_run_command.await_args.args[0]
        self.assertIn("send", args)
        self.assertIn("opencode attach http://x --session ses_1", " ".join(args))
        self.assertEqual(hook.identity_for("ses_1"), ("pane_1", "tab_1"))

    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_failure_is_swallowed(self, mock_run_command):
        mock_run_command.side_effect = RuntimeError("boom")

        hook = orca.OrcaVisibilityHook()
        await hook.after_spawn(("term_1", "pane_1", "tab_1"), "http://x", "ses_1")  # must not raise

        # Identity is still recorded even if the attach command itself failed.
        self.assertEqual(hook.identity_for("ses_1"), ("pane_1", "tab_1"))

    async def test_no_op_without_handle(self):
        hook = orca.OrcaVisibilityHook()
        await hook.after_spawn(None, "http://x", "ses_1")  # must not raise

        self.assertIsNone(hook.identity_for("ses_1"))


class OrcaVisibilityHookIsAliveTests(unittest.IsolatedAsyncioTestCase):
    async def test_true_when_no_handle(self):
        hook = orca.OrcaVisibilityHook()
        self.assertTrue(await hook.is_alive(None))

    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_delegates_to_orca_terminal_show(self, mock_run_command):
        mock_run_command.return_value = (0, "{}", "")
        hook = orca.OrcaVisibilityHook()

        self.assertTrue(await hook.is_alive(("term_1", "pane_1", "tab_1")))

        mock_run_command.return_value = (1, "", "not found")
        self.assertFalse(await hook.is_alive(("term_1", "pane_1", "tab_1")))


class OrcaVisibilityHookExtraEnvTests(unittest.TestCase):
    def test_with_handle_sets_identity_vars(self):
        with patch.dict(os.environ, {"ORCA_AGENT_LAUNCH_TOKEN": "tok", "ORCA_WORKTREE_ID": "wt"}):
            env = orca.OrcaVisibilityHook().extra_env(("term_1", "new_pane", "new_tab"))

        self.assertEqual(env["ORCA_PANE_KEY"], "new_pane")
        self.assertEqual(env["ORCA_TAB_ID"], "new_tab")
        self.assertEqual(env["ORCA_TERMINAL_HANDLE"], "term_1")
        self.assertEqual(env["ORCA_AGENT_LAUNCH_TOKEN"], "tok")
        self.assertEqual(env["ORCA_WORKTREE_ID"], "wt")

    def test_without_handle_returns_empty_dict(self):
        self.assertEqual(orca.OrcaVisibilityHook().extra_env(None), {})


class ReadOrcaHookEndpointTests(unittest.TestCase):
    def test_none_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(orca._read_orca_hook_endpoint())

    def test_falls_back_to_env_vars(self):
        with patch.dict(
            os.environ,
            {"ORCA_AGENT_HOOK_PORT": "1234", "ORCA_AGENT_HOOK_TOKEN": "tok"},
            clear=True,
        ):
            self.assertEqual(orca._read_orca_hook_endpoint(), ("1234", "tok", "", ""))

    def test_falls_back_to_env_vars_including_env_and_version(self):
        with patch.dict(
            os.environ,
            {
                "ORCA_AGENT_HOOK_PORT": "1234",
                "ORCA_AGENT_HOOK_TOKEN": "tok",
                "ORCA_AGENT_HOOK_ENV": "prod",
                "ORCA_AGENT_HOOK_VERSION": "1.2.3",
            },
            clear=True,
        ):
            self.assertEqual(
                orca._read_orca_hook_endpoint(), ("1234", "tok", "prod", "1.2.3")
            )

    def test_prefers_endpoint_file_over_env_vars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            endpoint_path = os.path.join(tmpdir, "endpoint.env")
            with open(endpoint_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "ORCA_AGENT_HOOK_PORT=5555\n"
                    "ORCA_AGENT_HOOK_TOKEN=file_tok\n"
                    "ORCA_AGENT_HOOK_ENV=file_env\n"
                    "ORCA_AGENT_HOOK_VERSION=9.9.9\n"
                )

            with patch.dict(
                os.environ,
                {
                    "ORCA_AGENT_HOOK_ENDPOINT": endpoint_path,
                    "ORCA_AGENT_HOOK_PORT": "1234",
                    "ORCA_AGENT_HOOK_TOKEN": "env_tok",
                },
                clear=True,
            ):
                self.assertEqual(
                    orca._read_orca_hook_endpoint(),
                    ("5555", "file_tok", "file_env", "9.9.9"),
                )

    def test_missing_endpoint_file_falls_back_to_env(self):
        with patch.dict(
            os.environ,
            {
                "ORCA_AGENT_HOOK_ENDPOINT": "/no/such/file",
                "ORCA_AGENT_HOOK_PORT": "1234",
                "ORCA_AGENT_HOOK_TOKEN": "tok",
            },
            clear=True,
        ):
            self.assertEqual(orca._read_orca_hook_endpoint(), ("1234", "tok", "", ""))


class OrcaStatusHookTests(unittest.IsolatedAsyncioTestCase):
    def _hook_with_identity(self, session_id: str, pane_key: str, tab_id: str) -> orca.OrcaStatusHook:
        visibility_hook = orca.OrcaVisibilityHook()
        visibility_hook._identity[session_id] = (pane_key, tab_id)
        return orca.OrcaStatusHook(visibility_hook)

    async def test_no_op_without_pane_identity(self):
        status_hook = orca.OrcaStatusHook(orca.OrcaVisibilityHook())

        with patch("server.orca.httpx2.AsyncClient") as mock_client_cls:
            await status_hook.post_status("ses_1", "SessionBusy", {"sessionID": "ses_1"})

        mock_client_cls.assert_not_called()

    async def test_no_op_without_hook_coords(self):
        status_hook = self._hook_with_identity("ses_1", "pane_1", "tab_1")

        with patch("server.orca._read_orca_hook_endpoint", return_value=None), \
             patch("server.orca.httpx2.AsyncClient") as mock_client_cls:
            await status_hook.post_status("ses_1", "SessionBusy", {"sessionID": "ses_1"})

        mock_client_cls.assert_not_called()

    async def test_posts_expected_body_and_headers(self):
        status_hook = self._hook_with_identity("ses_1", "pane_1", "tab_1")
        mock_client = make_mock_client(FakeResponse(json_data={}))

        with patch(
            "server.orca._read_orca_hook_endpoint", return_value=("9999", "tok_abc", "prod", "1.0.0")
        ), patch.dict(
            os.environ,
            {"ORCA_AGENT_LAUNCH_TOKEN": "launch_tok", "ORCA_WORKTREE_ID": "wt_1"},
        ), patch("server.orca.httpx2.AsyncClient", return_value=mock_client):
            await status_hook.post_status(
                "ses_1", "PermissionRequest", {"sessionID": "ses_1", "id": "perm_1"}
            )

        mock_client.post.assert_awaited_once_with(
            "http://127.0.0.1:9999/hook/opencode",
            json={
                "paneKey": "pane_1",
                "launchToken": "launch_tok",
                "tabId": "tab_1",
                "worktreeId": "wt_1",
                "env": "prod",
                "version": "1.0.0",
                "payload": {
                    "hook_event_name": "PermissionRequest",
                    "sessionID": "ses_1",
                    "id": "perm_1",
                },
            },
            headers={"X-Orca-Agent-Hook-Token": "tok_abc"},
        )

    async def test_post_failure_is_swallowed(self):
        status_hook = self._hook_with_identity("ses_1", "pane_1", "tab_1")

        with patch(
            "server.orca._read_orca_hook_endpoint", return_value=("9999", "tok_abc", "", "")
        ), patch("server.orca.httpx2.AsyncClient", side_effect=RuntimeError("boom")):
            await status_hook.post_status("ses_1", "SessionIdle", {})  # must not raise


class OrcaServerWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_registers_answer_permission_tool(self):
        tools = await orca.mcp.list_tools()
        self.assertIn("answer_permission", {tool.name for tool in tools})


if __name__ == "__main__":
    unittest.main()
