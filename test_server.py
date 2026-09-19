import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2

import server
from server import parse_json_output


class ParseJsonOutputTests(unittest.TestCase):
    def test_single_line_json(self):
        output = '{"ok": true, "value": 1}'
        self.assertEqual(parse_json_output(output), {"ok": True, "value": 1})

    def test_pretty_printed_multiline_json(self):
        output = """{
  "ok": true,
  "result": {
    "terminal": {
      "handle": "term_abc"
    }
  }
}"""
        result = parse_json_output(output)
        self.assertEqual(result["result"]["terminal"]["handle"], "term_abc")

    def test_fallback_finds_json_after_leading_banner_line(self):
        output = "some warning printed to stdout\n{\"ok\": true}"
        self.assertEqual(parse_json_output(output), {"ok": True})

    def test_fallback_picks_first_valid_object_when_two_are_present(self):
        # Documents the actual invariant: first-valid-dict-wins, not last.
        output = '{"first": 1}\n{"second": 2}'
        self.assertEqual(parse_json_output(output), {"first": 1})

    def test_pretty_printed_json_with_leading_banner(self):
        output = """Some banner/log line before the JSON
{
  "ok": true,
  "result": {
    "terminal": {
      "handle": "term_abc"
    }
  }
}"""
        result = parse_json_output(output)
        self.assertEqual(result["result"]["terminal"]["handle"], "term_abc")

    def test_pretty_printed_json_with_trailing_banner(self):
        output = """{
  "ok": true,
  "value": 42
}
trailing log noise after the JSON"""
        self.assertEqual(parse_json_output(output), {"ok": True, "value": 42})

    def test_unparsable_output_raises(self):
        with self.assertRaises(RuntimeError):
            parse_json_output("not json at all")

    def test_empty_output_raises(self):
        with self.assertRaises(RuntimeError):
            parse_json_output("")

    def test_non_dict_json_raises(self):
        with self.assertRaises(RuntimeError):
            parse_json_output("[1, 2, 3]")


class LoadBrowserConfigTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("server.get_project_dir", return_value=Path(tmp_dir)):
                with self.assertRaises(RuntimeError) as ctx:
                    server.load_browser_config()
                self.assertIn("not found", str(ctx.exception))

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "browser-review.yaml").write_text("accounts: [unclosed")
            with patch("server.get_project_dir", return_value=Path(tmp_dir)):
                with self.assertRaises(RuntimeError) as ctx:
                    server.load_browser_config()
                self.assertIn("Failed to read", str(ctx.exception))

    def test_non_dict_root_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "browser-review.yaml").write_text("- just\n- a\n- list\n")
            with patch("server.get_project_dir", return_value=Path(tmp_dir)):
                with self.assertRaises(RuntimeError) as ctx:
                    server.load_browser_config()
                self.assertIn("YAML object", str(ctx.exception))

    def test_valid_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "browser-review.yaml").write_text(
                "accounts:\n  demo:\n    username: u\n    password: p\n"
            )
            with patch("server.get_project_dir", return_value=Path(tmp_dir)):
                config = server.load_browser_config()
            self.assertEqual(config["accounts"]["demo"]["username"], "u")


class GetBrowserAccountTests(unittest.TestCase):
    def test_missing_accounts_key_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            server.get_browser_account({}, "demo")
        self.assertIn("accounts", str(ctx.exception))

    def test_accounts_not_a_dict_raises(self):
        with self.assertRaises(RuntimeError):
            server.get_browser_account({"accounts": []}, "demo")

    def test_unknown_account_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            server.get_browser_account({"accounts": {}}, "demo")
        self.assertIn("demo", str(ctx.exception))

    def test_missing_username_raises(self):
        config = {"accounts": {"demo": {"password": "p"}}}
        with self.assertRaises(RuntimeError):
            server.get_browser_account(config, "demo")

    def test_missing_password_raises(self):
        config = {"accounts": {"demo": {"username": "u"}}}
        with self.assertRaises(RuntimeError):
            server.get_browser_account(config, "demo")

    def test_valid_account_returns_credentials(self):
        config = {"accounts": {"demo": {"username": "u", "password": "p"}}}
        result = server.get_browser_account(config, "demo")
        self.assertEqual(result, {"username": "u", "password": "p"})


class RunCommandTests(unittest.IsolatedAsyncioTestCase):
    """Exercises real subprocesses (no mocking) since this is the foundation
    every Orca call goes through."""

    async def test_success_returns_code_and_output(self):
        returncode, stdout, stderr = await server.run_command(["echo", "hello"])
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout.strip(), "hello")
        self.assertEqual(stderr, "")

    async def test_nonzero_return_code(self):
        returncode, _, _ = await server.run_command(["sh", "-c", "exit 3"])
        self.assertEqual(returncode, 3)

    async def test_stderr_is_captured(self):
        _, _, stderr = await server.run_command(["sh", "-c", "echo oops 1>&2"])
        self.assertEqual(stderr.strip(), "oops")

    async def test_cwd_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, stdout, _ = await server.run_command(["pwd"], cwd=tmp_dir)
            self.assertEqual(
                os.path.realpath(stdout.strip()), os.path.realpath(tmp_dir)
            )

    async def test_timeout_raises_and_kills_process(self):
        with self.assertRaises(RuntimeError) as ctx:
            await server.run_command(["sleep", "5"], timeout=0.2)

        self.assertIn("timed out", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# OpenCode HTTP data channel.
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, json_data=None, text: str | None = None):
        self.status_code = status_code
        self._json_data = {} if json_data is None else json_data
        self.text = text if text is not None else json.dumps(self._json_data)
        self.is_error = status_code >= 400

    def json(self):
        return self._json_data


def make_mock_client(response: FakeResponse) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=response)
    return mock_client


class CreateOpencodeSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_and_returns_session_id(self):
        mock_client = make_mock_client(FakeResponse(json_data={"id": "ses_123"}))

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            session_id = await server.create_opencode_session("http://127.0.0.1:4096", title="t")

        self.assertEqual(session_id, "ses_123")
        mock_client.post.assert_awaited_once_with(
            "http://127.0.0.1:4096/session", json={"title": "t"}
        )

    async def test_error_status_raises(self):
        mock_client = make_mock_client(FakeResponse(status_code=400, text="bad request"))

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await server.create_opencode_session("http://x", title="t")

        self.assertIn("bad request", str(ctx.exception))

    async def test_missing_id_raises(self):
        mock_client = make_mock_client(FakeResponse(json_data={}))

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError):
                await server.create_opencode_session("http://x", title="t")

    async def test_non_dict_response_raises(self):
        mock_client = make_mock_client(FakeResponse(json_data=["not", "a", "dict"]))

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError):
                await server.create_opencode_session("http://x", title="t")


class SendOpencodePromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_expected_body_and_returns_json(self):
        payload = {"info": {"finish": "stop"}, "parts": [{"type": "text", "text": "hi"}]}
        mock_client = make_mock_client(FakeResponse(json_data=payload))

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            result = await server.send_opencode_prompt(
                "http://x", "ses_1", "reviewer", "do it", 30
            )

        self.assertEqual(result, payload)
        mock_client.post.assert_awaited_once_with(
            "http://x/session/ses_1/message",
            json={"agent": "reviewer", "parts": [{"type": "text", "text": "do it"}]},
        )

    async def test_error_status_raises_with_body(self):
        mock_client = make_mock_client(FakeResponse(status_code=500, text="boom"))

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await server.send_opencode_prompt("http://x", "ses_1", "reviewer", "do it", 30)

        self.assertIn("boom", str(ctx.exception))

    async def test_non_dict_response_raises(self):
        mock_client = make_mock_client(FakeResponse(json_data=["not", "a", "dict"]))

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError):
                await server.send_opencode_prompt("http://x", "ses_1", "reviewer", "do it", 30)


# ---------------------------------------------------------------------------
# Per-session opencode serve process + Orca pane management.
# ---------------------------------------------------------------------------


class FakeStdout:
    def __init__(self, lines=None, hang: bool = False):
        self._lines = list(lines or [])
        self._hang = hang

    async def readline(self) -> bytes:
        if self._hang:
            await asyncio.sleep(3600)
        if self._lines:
            return self._lines.pop(0)
        return b""


class FakeProcess:
    def __init__(self, lines=None, pid: int = 54321, returncode=None, hang: bool = False):
        self.stdout = FakeStdout(lines, hang=hang)
        self.pid = pid
        self.returncode = returncode

    async def wait(self):
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


class CreateSessionPaneTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_parses_full_pane_identity(self, mock_run_command):
        payload = json.dumps(
            {"result": {"terminal": {"handle": "term_1", "paneKey": "tab:leaf", "tabId": "tab"}}}
        )
        mock_run_command.return_value = (0, payload, "")

        pane = await server._create_session_pane("reviewer", Path("/tmp/p"))

        self.assertEqual(pane, ("term_1", "tab:leaf", "tab"))

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_returns_none_on_command_failure(self, mock_run_command):
        mock_run_command.return_value = (1, "", "orca not running")

        pane = await server._create_session_pane("reviewer", Path("/tmp/p"))

        self.assertIsNone(pane)

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_returns_none_on_incomplete_identity(self, mock_run_command):
        payload = json.dumps({"result": {"terminal": {"handle": "term_1"}}})
        mock_run_command.return_value = (0, payload, "")

        pane = await server._create_session_pane("reviewer", Path("/tmp/p"))

        self.assertIsNone(pane)

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_returns_none_on_exception(self, mock_run_command):
        mock_run_command.side_effect = RuntimeError("boom")

        pane = await server._create_session_pane("reviewer", Path("/tmp/p"))

        self.assertIsNone(pane)


class SessionServerEnvTests(unittest.TestCase):
    def test_with_pane_overrides_identity_vars(self):
        with patch.dict(os.environ, {"ORCA_PANE_KEY": "old", "ORCA_TAB_ID": "old", "OTHER": "z"}):
            env = server._session_server_env(("term_1", "new_pane", "new_tab"))

        self.assertEqual(env["ORCA_PANE_KEY"], "new_pane")
        self.assertEqual(env["ORCA_TAB_ID"], "new_tab")
        self.assertEqual(env["ORCA_TERMINAL_HANDLE"], "term_1")
        self.assertEqual(env.get("OTHER"), "z")

    def test_without_pane_strips_identity_vars(self):
        with patch.dict(
            os.environ,
            {
                "ORCA_PANE_KEY": "old",
                "ORCA_TAB_ID": "old",
                "ORCA_TERMINAL_HANDLE": "old",
                "OTHER": "z",
            },
        ):
            env = server._session_server_env(None)

        self.assertNotIn("ORCA_PANE_KEY", env)
        self.assertNotIn("ORCA_TAB_ID", env)
        self.assertNotIn("ORCA_TERMINAL_HANDLE", env)
        self.assertEqual(env.get("OTHER"), "z")

    def test_leaves_hook_infrastructure_vars_untouched(self):
        with patch.dict(
            os.environ, {"ORCA_OPENCODE_CONFIG_DIR": "/x", "ORCA_AGENT_HOOK_PORT": "1"}
        ):
            env_with = server._session_server_env(("h", "p", "t"))
            env_without = server._session_server_env(None)

        self.assertEqual(env_with.get("ORCA_OPENCODE_CONFIG_DIR"), "/x")
        self.assertEqual(env_without.get("ORCA_OPENCODE_CONFIG_DIR"), "/x")
        self.assertEqual(env_with.get("ORCA_AGENT_HOOK_PORT"), "1")


class SpawnSessionServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_process_and_base_url(self):
        process = FakeProcess([b"opencode server listening on http://127.0.0.1:4096\n"])

        with patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            result_process, base_url = await server._spawn_session_server(Path("/tmp/p"), None)

        self.assertIs(result_process, process)
        self.assertEqual(base_url, "http://127.0.0.1:4096")

    @patch("server.os.killpg")
    @patch("server.os.getpgid", return_value=1)
    async def test_process_exits_without_listening_line_raises(self, mock_getpgid, mock_killpg):
        process = FakeProcess([b"some startup noise\n"], returncode=1)

        with patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            with self.assertRaises(RuntimeError):
                await server._spawn_session_server(Path("/tmp/p"), None)

    @patch("server.os.killpg")
    @patch("server.os.getpgid", return_value=1)
    async def test_exit_code_in_error_reflects_post_wait_value(self, mock_getpgid, mock_killpg):
        # returncode starts as None (as it would be immediately after EOF,
        # before asyncio has necessarily updated it) and FakeProcess.wait()
        # is what finalizes it to -9 -- the error message must reflect that
        # finalized value, not the pre-wait None.
        process = FakeProcess([b"no listening line here\n"], returncode=None)

        with patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            with self.assertRaises(RuntimeError) as ctx:
                await server._spawn_session_server(Path("/tmp/p"), None)

        self.assertIn("-9", str(ctx.exception))
        self.assertNotIn("None", str(ctx.exception))

    @patch("server.os.killpg")
    @patch("server.os.getpgid", return_value=1)
    @patch("server.OPENCODE_SERVE_STARTUP_TIMEOUT_SECONDS", 0.05)
    async def test_timeout_raises_and_kills_process(self, mock_getpgid, mock_killpg):
        process = FakeProcess(hang=True)

        with patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            with self.assertRaises(asyncio.TimeoutError):
                await server._spawn_session_server(Path("/tmp/p"), None)

        mock_killpg.assert_called_once()

    async def test_passes_pane_env_to_subprocess(self):
        process = FakeProcess([b"opencode server listening on http://127.0.0.1:4096\n"])
        mock_exec = AsyncMock(return_value=process)

        with patch("server.asyncio.create_subprocess_exec", new=mock_exec):
            await server._spawn_session_server(Path("/tmp/p"), ("term_1", "pane_1", "tab_1"))

        _args, kwargs = mock_exec.call_args
        self.assertEqual(kwargs["env"]["ORCA_PANE_KEY"], "pane_1")


class AttachTerminalToSessionTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_sends_attach_command(self, mock_run_command):
        mock_run_command.return_value = (0, "{}", "")

        await server._attach_terminal_to_session("term_1", "http://x", "ses_1")

        args = mock_run_command.await_args.args[0]
        self.assertIn("send", args)
        self.assertIn("opencode attach http://x --session ses_1", " ".join(args))

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_failure_is_swallowed(self, mock_run_command):
        mock_run_command.side_effect = RuntimeError("boom")

        await server._attach_terminal_to_session("term_1", "http://x", "ses_1")  # must not raise


class CreateSessionBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._session_state.clear()

    @patch("server._attach_terminal_to_session", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server._spawn_session_server", new_callable=AsyncMock)
    @patch("server._create_session_pane", new_callable=AsyncMock)
    async def test_new_session_creates_pane_server_and_session(
        self, mock_create_pane, mock_spawn, mock_create_session, mock_attach
    ):
        process = FakeProcess([])
        mock_create_pane.return_value = ("term_1", "pane_1", "tab_1")
        mock_spawn.return_value = (process, "http://x")
        mock_create_session.return_value = "ses_new"

        base_url, session_id = await server._create_session_backend(
            None, "reviewer", Path("/tmp/p")
        )

        self.assertEqual((base_url, session_id), ("http://x", "ses_new"))
        mock_spawn.assert_awaited_once_with(Path("/tmp/p"), ("term_1", "pane_1", "tab_1"))
        mock_attach.assert_awaited_once_with("term_1", "http://x", "ses_new")
        self.assertEqual(server._session_state["ses_new"], (process, "http://x", "term_1"))

    @patch("server._attach_terminal_to_session", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server._spawn_session_server", new_callable=AsyncMock)
    @patch("server._create_session_pane", new_callable=AsyncMock)
    async def test_falls_back_to_headless_when_orca_unavailable(
        self, mock_create_pane, mock_spawn, mock_create_session, mock_attach
    ):
        process = FakeProcess([])
        mock_create_pane.return_value = None
        mock_spawn.return_value = (process, "http://x")
        mock_create_session.return_value = "ses_new"

        await server._create_session_backend(None, "reviewer", Path("/tmp/p"))

        mock_spawn.assert_awaited_once_with(Path("/tmp/p"), None)
        mock_attach.assert_not_awaited()
        self.assertEqual(server._session_state["ses_new"], (process, "http://x", None))

    @patch("server._attach_terminal_to_session", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server._spawn_session_server", new_callable=AsyncMock)
    @patch("server._create_session_pane", new_callable=AsyncMock)
    async def test_existing_session_id_skips_create_opencode_session(
        self, mock_create_pane, mock_spawn, mock_create_session, mock_attach
    ):
        process = FakeProcess([])
        mock_create_pane.return_value = None
        mock_spawn.return_value = (process, "http://x")

        base_url, session_id = await server._create_session_backend(
            "ses_existing", "reviewer", Path("/tmp/p")
        )

        self.assertEqual(session_id, "ses_existing")
        mock_create_session.assert_not_awaited()


class EnsureSessionBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._session_state.clear()

    @patch("server._create_session_backend", new_callable=AsyncMock)
    async def test_new_session_always_creates(self, mock_create_backend):
        mock_create_backend.return_value = ("http://x", "ses_new")

        result = await server.ensure_session_backend(None, "reviewer", Path("/tmp/p"))

        self.assertEqual(result, ("http://x", "ses_new"))
        mock_create_backend.assert_awaited_once_with(None, "reviewer", Path("/tmp/p"))

    @patch("server.run_command", new_callable=AsyncMock)
    @patch("server._create_session_backend", new_callable=AsyncMock)
    async def test_reuses_when_server_and_terminal_alive(self, mock_create_backend, mock_run_command):
        process = FakeProcess([], returncode=None)
        server._session_state["ses_1"] = (process, "http://x", "term_1")
        mock_run_command.return_value = (0, "{}", "")

        result = await server.ensure_session_backend("ses_1", "reviewer", Path("/tmp/p"))

        self.assertEqual(result, ("http://x", "ses_1"))
        mock_create_backend.assert_not_awaited()

    @patch("server.run_command", new_callable=AsyncMock)
    @patch("server._create_session_backend", new_callable=AsyncMock)
    async def test_recreates_when_server_dead(self, mock_create_backend, mock_run_command):
        dead_process = FakeProcess([], returncode=0)
        server._session_state["ses_1"] = (dead_process, "http://x", "term_1")
        mock_create_backend.return_value = ("http://y", "ses_1")

        result = await server.ensure_session_backend("ses_1", "reviewer", Path("/tmp/p"))

        self.assertEqual(result, ("http://y", "ses_1"))
        mock_create_backend.assert_awaited_once_with("ses_1", "reviewer", Path("/tmp/p"))
        mock_run_command.assert_not_awaited()  # server already dead, no need to check the terminal

    @patch("server.run_command", new_callable=AsyncMock)
    @patch("server._create_session_backend", new_callable=AsyncMock)
    async def test_recreates_when_terminal_gone(self, mock_create_backend, mock_run_command):
        process = FakeProcess([], returncode=None)
        server._session_state["ses_1"] = (process, "http://x", "term_1")
        mock_run_command.return_value = (1, "", "not found")
        mock_create_backend.return_value = ("http://y", "ses_1")

        result = await server.ensure_session_backend("ses_1", "reviewer", Path("/tmp/p"))

        self.assertEqual(result, ("http://y", "ses_1"))
        mock_create_backend.assert_awaited_once_with("ses_1", "reviewer", Path("/tmp/p"))

    @patch("server._create_session_backend", new_callable=AsyncMock)
    async def test_concurrent_calls_do_not_double_create(self, mock_create_backend):
        call_count = 0

        async def fake_create_backend(session_id, agent, project_dir):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            server._session_state[session_id] = (FakeProcess([], returncode=None), "http://x", None)
            return "http://x", session_id

        mock_create_backend.side_effect = fake_create_backend

        await asyncio.gather(
            server.ensure_session_backend("ses_concurrent", "reviewer", Path("/tmp/p")),
            server.ensure_session_backend("ses_concurrent", "reviewer", Path("/tmp/p")),
        )

        self.assertEqual(call_count, 1)


class KillOpencodeServersTests(unittest.TestCase):
    def setUp(self):
        server._session_state.clear()

    @patch("server.os.killpg")
    @patch("server.os.getpgid", return_value=42)
    def test_kills_live_processes_only(self, mock_getpgid, mock_killpg):
        live = FakeProcess([], pid=111, returncode=None)
        dead = FakeProcess([], pid=222, returncode=0)
        server._session_state["ses_a"] = (live, "http://x", "term_a")
        server._session_state["ses_b"] = (dead, "http://y", "term_b")

        server._kill_opencode_servers()

        mock_getpgid.assert_called_once_with(111)
        mock_killpg.assert_called_once_with(42, server.signal.SIGTERM)

    @patch("server.os.killpg", side_effect=[Exception("boom"), None])
    @patch("server.os.getpgid", return_value=42)
    def test_exception_on_one_does_not_block_others(self, mock_getpgid, mock_killpg):
        live1 = FakeProcess([], pid=111, returncode=None)
        live2 = FakeProcess([], pid=222, returncode=None)
        server._session_state["ses_c"] = (live1, "http://x", None)
        server._session_state["ses_d"] = (live2, "http://y", None)

        server._kill_opencode_servers()  # must not raise

        self.assertEqual(mock_killpg.call_count, 2)


class InstallTerminationCleanupTests(unittest.TestCase):
    def test_chains_to_previous_callable_handler(self):
        previous = MagicMock()
        installed_handlers = {}

        with patch("server.signal.getsignal", return_value=previous), \
             patch("server.signal.signal", side_effect=lambda sig, h: installed_handlers.__setitem__(sig, h)), \
             patch("server._kill_opencode_servers") as mock_kill:
            server._install_termination_cleanup()

            self.assertIn(server.signal.SIGTERM, installed_handlers)
            installed_handlers[server.signal.SIGTERM](server.signal.SIGTERM, None)

            mock_kill.assert_called_once()
        previous.assert_called_once_with(server.signal.SIGTERM, None)

    def test_falls_back_to_default_when_previous_is_sig_dfl(self):
        installed_handlers = {}

        with patch("server.signal.getsignal", return_value=server.signal.SIG_DFL), \
             patch("server.signal.signal", side_effect=lambda sig, h: installed_handlers.__setitem__(sig, h)), \
             patch("server._kill_opencode_servers") as mock_kill, \
             patch("server.os.kill") as mock_os_kill:
            server._install_termination_cleanup()
            installed_handlers[server.signal.SIGTERM](server.signal.SIGTERM, None)

            mock_kill.assert_called_once()
            mock_os_kill.assert_called_once()

    def test_leaves_sig_ign_untouched(self):
        installed_handlers = {}

        with patch("server.signal.getsignal", return_value=server.signal.SIG_IGN), \
             patch("server.signal.signal", side_effect=lambda sig, h: installed_handlers.__setitem__(sig, h)), \
             patch("server._kill_opencode_servers") as mock_kill, \
             patch("server.os.kill") as mock_os_kill:
            server._install_termination_cleanup()
            installed_handlers[server.signal.SIGTERM](server.signal.SIGTERM, None)

            mock_kill.assert_called_once()
            mock_os_kill.assert_not_called()


# ---------------------------------------------------------------------------
# run_opencode orchestration.
# ---------------------------------------------------------------------------


class RunOpencodeTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_creates_new_session_when_none_given(
        self, mock_ensure_backend, mock_send_prompt
    ):
        mock_ensure_backend.return_value = ("http://127.0.0.1:4096", "ses_new")
        mock_send_prompt.return_value = {
            "info": {"finish": "stop"},
            "parts": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}],
        }

        result = await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(
            result, {"session_id": "ses_new", "response": "hello world", "finish_reason": "stop"}
        )
        mock_ensure_backend.assert_awaited_once_with(None, "reviewer", server.get_project_dir())
        mock_send_prompt.assert_awaited_once_with(
            "http://127.0.0.1:4096", "ses_new", "reviewer", "do it",
            server.OPENCODE_PROMPT_TIMEOUT_SECONDS,
        )

    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_reuses_given_session_id(self, mock_ensure_backend, mock_send_prompt):
        mock_ensure_backend.return_value = ("http://x", "ses_existing")
        mock_send_prompt.return_value = {"info": {"finish": "stop"}, "parts": []}

        result = await server.run_opencode(agent="reviewer", prompt="do it", session_id="ses_existing")

        self.assertEqual(result["session_id"], "ses_existing")
        mock_ensure_backend.assert_awaited_once_with(
            "ses_existing", "reviewer", server.get_project_dir()
        )

    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_error_in_info_raises(self, mock_ensure_backend, mock_send_prompt):
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.return_value = {"info": {"error": {"message": "boom"}}, "parts": []}

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertIn("boom", str(ctx.exception))

    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_non_text_parts_are_ignored(self, mock_ensure_backend, mock_send_prompt):
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.return_value = {
            "info": {"finish": "stop"},
            "parts": [
                {"type": "step-start"},
                {"type": "text", "text": "kept"},
                {"type": "tool", "text": "ignored-because-not-text-type"},
            ],
        }

        result = await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(result["response"], "kept")

    @patch("server.os.killpg")
    @patch("server.os.getpgid", return_value=1)
    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_retries_once_after_transport_error(
        self, mock_ensure_backend, mock_send_prompt, mock_getpgid, mock_killpg
    ):
        mock_ensure_backend.side_effect = [
            ("http://first", "ses_new"),
            ("http://second", "ses_new"),
        ]
        mock_send_prompt.side_effect = [
            httpx2.ConnectError("boom"),
            {"info": {"finish": "stop"}, "parts": [{"type": "text", "text": "ok"}]},
        ]
        stale_process = FakeProcess([], pid=999, returncode=None)
        server._session_state["ses_new"] = (stale_process, "http://first", None)

        result = await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(result["response"], "ok")
        self.assertEqual(mock_ensure_backend.await_count, 2)
        self.assertEqual(mock_send_prompt.await_count, 2)
        self.assertNotIn("ses_new", server._session_state)
        # The stale server must actually be killed, not just forgotten --
        # otherwise it leaks as an untracked, unkillable orphan.
        mock_killpg.assert_called_once_with(1, server.signal.SIGTERM)
        # The second attempt must reuse the session_id learned from the first
        # (not start over with session_id=None), so history isn't lost.
        second_call_args = mock_ensure_backend.await_args_list[1].args
        self.assertEqual(second_call_args[0], "ses_new")

    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_read_timeout_does_not_trigger_retry(
        self, mock_ensure_backend, mock_send_prompt
    ):
        # A slow-but-connected call must not be torn down and retried just
        # because it took a while -- only actual connection failures should.
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.side_effect = httpx2.ReadTimeout("slow")

        with self.assertRaises(httpx2.ReadTimeout):
            await server.run_opencode(agent="reviewer", prompt="do it")

        mock_send_prompt.assert_awaited_once()

    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_raises_after_second_transport_error(
        self, mock_ensure_backend, mock_send_prompt
    ):
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.side_effect = httpx2.ConnectError("boom")

        with self.assertRaises(httpx2.TransportError):
            await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(mock_send_prompt.await_count, 2)

    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_retry_reuses_explicit_session_id(
        self, mock_ensure_backend, mock_send_prompt
    ):
        mock_ensure_backend.return_value = ("http://x", "ses_existing")
        mock_send_prompt.side_effect = [
            httpx2.ConnectError("boom"),
            {"info": {"finish": "stop"}, "parts": []},
        ]

        result = await server.run_opencode(agent="reviewer", prompt="do it", session_id="ses_existing")

        self.assertEqual(result["session_id"], "ses_existing")
        for call in mock_ensure_backend.await_args_list:
            self.assertEqual(call.args[0], "ses_existing")


class RegisterAgentToolTests(unittest.IsolatedAsyncioTestCase):
    """
    _register_agent_tool() builds the handler that becomes a dynamically
    registered MCP tool for one agent. Tested by capturing the handler
    mcp.add_tool() would otherwise receive, instead of exercising the real
    tool registry (which already owns "implementer"/"reviewer" from the
    module-level discovery loop and would reject a same-name re-registration).
    """

    def _capture_handler(self, name: str, description: str):
        captured = {}

        def fake_add_tool(fn, name=None, description=None):
            captured["fn"] = fn
            captured["name"] = name
            captured["description"] = description

        with patch.object(server.mcp, "add_tool", side_effect=fake_add_tool):
            server._register_agent_tool(name, description)

        return captured

    async def test_success_calls_run_opencode_with_agent_name(self):
        captured = self._capture_handler("implementer", "Implements stuff")
        self.assertEqual(captured["name"], "implementer")
        self.assertEqual(captured["description"], "Implements stuff")

        with patch("server.run_opencode", new_callable=AsyncMock) as mock_run_opencode:
            mock_run_opencode.return_value = {
                "session_id": "ses_1",
                "response": "done",
                "finish_reason": "stop",
            }
            result = await captured["fn"](plan="do this", session_id="ses_1")

        self.assertEqual(result["response"], "done")
        call_kwargs = mock_run_opencode.await_args.kwargs
        self.assertEqual(call_kwargs["agent"], "implementer")
        self.assertEqual(call_kwargs["session_id"], "ses_1")
        self.assertEqual(call_kwargs["prompt"], "do this")

    async def test_failure_converts_to_tool_error(self):
        captured = self._capture_handler("reviewer", "Reviews stuff")

        with patch("server.run_opencode", new_callable=AsyncMock) as mock_run_opencode:
            mock_run_opencode.side_effect = RuntimeError("underlying failure")
            with self.assertRaises(server.ToolError) as ctx:
                await captured["fn"](plan="check this")

        self.assertIn("underlying failure", str(ctx.exception))

    async def test_review_browser_raises_tool_error(self):
        with self.assertRaises(server.ToolError):
            await server.review_browser(test_plan="x", account="demo")


class DiscoverAgentsTests(unittest.TestCase):
    @staticmethod
    def _write_agent(directory: Path, slug: str, frontmatter_body: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{slug}.md").write_text(
            f"---\n{frontmatter_body}\n---\n\nSome agent instructions.\n",
            encoding="utf-8",
        )

    def _discover(self, home: Path, project_dir: Path) -> dict:
        with patch.object(Path, "home", return_value=home):
            return server.discover_agents(project_dir)

    def test_discovers_global_agent_with_description(self):
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as project_dir:
            home = Path(home_dir)
            self._write_agent(
                home / ".config" / "opencode" / "agent", "coder", "description: Writes code"
            )

            agents = self._discover(home, Path(project_dir))

        self.assertEqual(agents, {"coder": "Writes code"})

    def test_missing_description_falls_back_to_generic_text(self):
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as project_dir:
            home = Path(home_dir)
            self._write_agent(home / ".config" / "opencode" / "agent", "coder", "mode: primary")

            agents = self._discover(home, Path(project_dir))

        self.assertEqual(agents, {"coder": "Run the 'coder' OpenCode agent"})

    def test_malformed_closing_delimiter_is_skipped(self):
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as project_dir:
            home = Path(home_dir)
            agent_dir = home / ".config" / "opencode" / "agent"
            agent_dir.mkdir(parents=True)
            (agent_dir / "review-browser.md").write_text(
                "---\ndescription: Broken\n-----------------------\n\nBody.\n",
                encoding="utf-8",
            )

            agents = self._discover(home, Path(project_dir))

        self.assertEqual(agents, {})

    def test_invalid_yaml_is_skipped(self):
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as project_dir:
            home = Path(home_dir)
            self._write_agent(
                home / ".config" / "opencode" / "agent", "broken", "description: [unterminated"
            )

            agents = self._discover(home, Path(project_dir))

        self.assertEqual(agents, {})

    def test_no_frontmatter_is_skipped(self):
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as project_dir:
            home = Path(home_dir)
            agent_dir = home / ".config" / "opencode" / "agent"
            agent_dir.mkdir(parents=True)
            (agent_dir / "plain.md").write_text("Just a plain agent file.\n", encoding="utf-8")

            agents = self._discover(home, Path(project_dir))

        self.assertEqual(agents, {})

    def test_project_agent_overrides_global_of_same_name(self):
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as project_dir:
            home = Path(home_dir)
            project_dir_path = Path(project_dir)
            self._write_agent(
                home / ".config" / "opencode" / "agent", "reviewer", "description: Global reviewer"
            )
            self._write_agent(
                project_dir_path / ".opencode" / "agent", "reviewer", "description: Project reviewer"
            )

            agents = self._discover(home, project_dir_path)

        self.assertEqual(agents, {"reviewer": "Project reviewer"})

    def test_missing_agent_directories_return_empty(self):
        with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as project_dir:
            agents = self._discover(Path(home_dir), Path(project_dir) / "does-not-exist")

        self.assertEqual(agents, {})


if __name__ == "__main__":
    unittest.main()
