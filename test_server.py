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


class SandboxExecEnvTests(unittest.TestCase):
    def test_with_pane_sets_identity_vars(self):
        with patch.dict(os.environ, {"ORCA_AGENT_LAUNCH_TOKEN": "tok", "ORCA_WORKTREE_ID": "wt"}):
            env = server._sandbox_exec_env(("term_1", "new_pane", "new_tab"))

        self.assertEqual(env["ORCA_PANE_KEY"], "new_pane")
        self.assertEqual(env["ORCA_TAB_ID"], "new_tab")
        self.assertEqual(env["ORCA_TERMINAL_HANDLE"], "term_1")
        self.assertEqual(env["ORCA_AGENT_LAUNCH_TOKEN"], "tok")
        self.assertEqual(env["ORCA_WORKTREE_ID"], "wt")

    def test_without_pane_omits_identity_vars(self):
        env = server._sandbox_exec_env(None)

        self.assertNotIn("ORCA_PANE_KEY", env)
        self.assertNotIn("ORCA_TAB_ID", env)
        self.assertNotIn("ORCA_TERMINAL_HANDLE", env)
        self.assertNotIn("ORCA_AGENT_HOOK_PORT", env)

    def test_forwards_hooks_dir_regardless_of_pane(self):
        with patch.dict(os.environ, {"OPENCODE_CONFIG_DIR": "/x"}):
            env_with = server._sandbox_exec_env(("h", "p", "t"))
            env_without = server._sandbox_exec_env(None)

        self.assertEqual(env_with.get("OPENCODE_CONFIG_DIR"), "/x")
        self.assertEqual(env_without.get("OPENCODE_CONFIG_DIR"), "/x")


class ReadOrcaHookEndpointTests(unittest.TestCase):
    def test_none_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(server._read_orca_hook_endpoint())

    def test_falls_back_to_env_vars(self):
        with patch.dict(
            os.environ,
            {"ORCA_AGENT_HOOK_PORT": "1234", "ORCA_AGENT_HOOK_TOKEN": "tok"},
            clear=True,
        ):
            self.assertEqual(server._read_orca_hook_endpoint(), ("1234", "tok", "", ""))

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
                server._read_orca_hook_endpoint(), ("1234", "tok", "prod", "1.2.3")
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
                    server._read_orca_hook_endpoint(),
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
            self.assertEqual(server._read_orca_hook_endpoint(), ("1234", "tok", "", ""))


class PostOrcaStatusHookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._session_hook_identity.clear()

    def tearDown(self):
        server._session_hook_identity.clear()

    async def test_no_op_without_pane_identity(self):
        with patch("server.httpx2.AsyncClient") as mock_client_cls:
            await server._post_orca_status_hook("ses_1", "SessionBusy", {"sessionID": "ses_1"})

        mock_client_cls.assert_not_called()

    async def test_no_op_without_hook_coords(self):
        server._session_hook_identity["ses_1"] = ("pane_1", "tab_1")

        with patch("server._read_orca_hook_endpoint", return_value=None), \
             patch("server.httpx2.AsyncClient") as mock_client_cls:
            await server._post_orca_status_hook("ses_1", "SessionBusy", {"sessionID": "ses_1"})

        mock_client_cls.assert_not_called()

    async def test_posts_expected_body_and_headers(self):
        server._session_hook_identity["ses_1"] = ("pane_1", "tab_1")
        mock_client = make_mock_client(FakeResponse(json_data={}))

        with patch(
            "server._read_orca_hook_endpoint", return_value=("9999", "tok_abc", "prod", "1.0.0")
        ), patch.dict(
            os.environ,
            {"ORCA_AGENT_LAUNCH_TOKEN": "launch_tok", "ORCA_WORKTREE_ID": "wt_1"},
        ), patch("server.httpx2.AsyncClient", return_value=mock_client):
            await server._post_orca_status_hook(
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
        server._session_hook_identity["ses_1"] = ("pane_1", "tab_1")

        with patch(
            "server._read_orca_hook_endpoint", return_value=("9999", "tok_abc", "", "")
        ), patch("server.httpx2.AsyncClient", side_effect=RuntimeError("boom")):
            await server._post_orca_status_hook("ses_1", "SessionIdle", {})  # must not raise


class GetAgentNetworkProfileTests(unittest.TestCase):
    def test_defaults_to_balanced_when_no_definition_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = server._get_agent_network_profile("nope", Path(tmpdir))

        self.assertEqual(profile, "balanced")

    def test_reads_declared_profile_from_project_local_definition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            agent_dir = project_dir / ".opencode" / "agent"
            agent_dir.mkdir(parents=True)
            (agent_dir / "reviewer.md").write_text(
                "---\ndescription: x\nnetwork: none\n---\nbody", encoding="utf-8"
            )

            profile = server._get_agent_network_profile("reviewer", project_dir)

        self.assertEqual(profile, "none")

    def test_invalid_value_falls_back_to_balanced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            agent_dir = project_dir / ".opencode" / "agent"
            agent_dir.mkdir(parents=True)
            (agent_dir / "reviewer.md").write_text(
                "---\ndescription: x\nnetwork: wide-open\n---\nbody", encoding="utf-8"
            )

            profile = server._get_agent_network_profile("reviewer", project_dir)

        self.assertEqual(profile, "balanced")


class InjectAgentDefinitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipes_content_over_stdin_to_sbx_exec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "reviewer.md"
            source_path.write_text("---\ndescription: x\n---\nbody", encoding="utf-8")

            process = AsyncMock()
            process.communicate = AsyncMock(return_value=(b"", b""))
            process.returncode = 0

            with patch(
                "server.asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = process
                await server._inject_agent_definition("sbx_1", "reviewer", source_path)

            args = mock_exec.await_args.args
            self.assertEqual(args[:4], ("sbx", "exec", "-i", "sbx_1"))
            self.assertIn("reviewer.md", args[-1])
            process.communicate.assert_awaited_once_with(
                input="---\ndescription: x\n---\nbody".encode("utf-8")
            )

    async def test_nonzero_exit_is_logged_not_raised(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "reviewer.md"
            source_path.write_text("body", encoding="utf-8")

            process = AsyncMock()
            process.communicate = AsyncMock(return_value=(b"", b"permission denied"))
            process.returncode = 1

            with patch(
                "server.asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = process
                await server._inject_agent_definition("sbx_1", "reviewer", source_path)  # must not raise

    async def test_exception_is_swallowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "reviewer.md"
            source_path.write_text("body", encoding="utf-8")

            with patch(
                "server.asyncio.create_subprocess_exec", side_effect=OSError("boom")
            ):
                await server._inject_agent_definition("sbx_1", "reviewer", source_path)  # must not raise


class SbxInspectTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_finds_matching_sandbox(self, mock_run_command):
        mock_run_command.return_value = (
            0,
            json.dumps({"sandboxes": [{"name": "orca-opencode-abc", "status": "running"}]}),
            "",
        )

        sandbox = await server._sbx_inspect("orca-opencode-abc")

        self.assertEqual(sandbox["status"], "running")

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_returns_none_when_not_found(self, mock_run_command):
        mock_run_command.return_value = (0, json.dumps({"sandboxes": []}), "")

        self.assertIsNone(await server._sbx_inspect("missing"))

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_raises_on_command_failure(self, mock_run_command):
        mock_run_command.return_value = (1, "", "daemon not running")

        with self.assertRaises(RuntimeError):
            await server._sbx_inspect("any")

    @patch("server._sbx_inspect", new_callable=AsyncMock)
    async def test_published_port_matches_by_sandbox_port(self, mock_inspect):
        mock_inspect.return_value = {
            "ports": [{"sandbox_port": 4096, "host_port": 49152}]
        }

        port = await server._sbx_published_port("orca-opencode-abc", 4096)

        self.assertEqual(port, 49152)

    @patch("server._sbx_inspect", new_callable=AsyncMock)
    async def test_published_port_none_when_sandbox_missing(self, mock_inspect):
        mock_inspect.return_value = None

        self.assertIsNone(await server._sbx_published_port("missing", 4096))


class SpawnSessionServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch("server._find_agent_definition", return_value=None)
        self.addCleanup(patcher.stop)
        patcher.start()
        patcher = patch("server._inject_agent_definition", new_callable=AsyncMock)
        self.addCleanup(patcher.stop)
        patcher.start()
        patcher = patch("server._get_agent_network_profile", return_value="balanced")
        self.addCleanup(patcher.stop)
        patcher.start()

    async def test_injects_global_agent_definition_before_starting_serve(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])
        global_path = Path("/home/host-user/.config/opencode/agent/reviewer.md")

        with patch("server.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("server._sbx_published_port", new_callable=AsyncMock, return_value=49152), \
             patch("server._find_agent_definition", return_value=global_path), \
             patch("server._inject_agent_definition", new_callable=AsyncMock) as mock_inject:
            mock_run_command.return_value = (0, "", "")

            _process, _base_url, sandbox_name = await server._spawn_session_server(
                Path("/tmp/p"), None, "reviewer"
            )

        mock_inject.assert_awaited_once_with(sandbox_name, "reviewer", global_path)

    async def test_skips_injection_for_project_local_definition(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])
        project_local_path = Path("/tmp/p/.opencode/agent/reviewer.md")

        with patch("server.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("server._sbx_published_port", new_callable=AsyncMock, return_value=49152), \
             patch("server._find_agent_definition", return_value=project_local_path), \
             patch("server._inject_agent_definition", new_callable=AsyncMock) as mock_inject:
            mock_run_command.return_value = (0, "", "")

            await server._spawn_session_server(Path("/tmp/p"), None, "reviewer")

        mock_inject.assert_not_awaited()

    async def test_returns_process_base_url_and_sandbox_name(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])

        with patch("server.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("server._sbx_published_port", new_callable=AsyncMock) as mock_port:
            mock_run_command.return_value = (0, "", "")
            mock_port.return_value = 49152

            result_process, base_url, sandbox_name = await server._spawn_session_server(
                Path("/tmp/p"), None, "reviewer"
            )

        self.assertIs(result_process, process)
        self.assertEqual(base_url, "http://127.0.0.1:49152")
        self.assertTrue(sandbox_name.startswith("orca-opencode-"))

    async def test_sbx_create_failure_raises(self):
        with patch("server.run_command", new_callable=AsyncMock) as mock_run_command:
            mock_run_command.return_value = (1, "", "sbx: image pull failed")

            with self.assertRaises(RuntimeError):
                await server._spawn_session_server(Path("/tmp/p"), None, "reviewer")

    @patch("server.os.killpg")
    @patch("server.os.getpgid", return_value=1)
    async def test_process_exits_without_listening_line_raises_and_removes_sandbox(
        self, mock_getpgid, mock_killpg
    ):
        process = FakeProcess([b"some startup noise\n"], returncode=1)

        with patch("server.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("server._sbx_remove_sandbox", new_callable=AsyncMock) as mock_remove:
            mock_run_command.return_value = (0, "", "")

            with self.assertRaises(RuntimeError):
                await server._spawn_session_server(Path("/tmp/p"), None, "reviewer")

        mock_remove.assert_awaited_once()

    @patch("server.os.killpg")
    @patch("server.os.getpgid", return_value=1)
    @patch("server.OPENCODE_SERVE_STARTUP_TIMEOUT_SECONDS", 0.05)
    async def test_timeout_raises_kills_process_and_removes_sandbox(
        self, mock_getpgid, mock_killpg
    ):
        process = FakeProcess(hang=True)

        with patch("server.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("server._sbx_remove_sandbox", new_callable=AsyncMock) as mock_remove:
            mock_run_command.return_value = (0, "", "")

            with self.assertRaises(asyncio.TimeoutError):
                await server._spawn_session_server(Path("/tmp/p"), None, "reviewer")

        mock_killpg.assert_called_once()
        mock_remove.assert_awaited_once()

    async def test_published_port_missing_raises_and_removes_sandbox(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])

        with patch("server.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("server._sbx_published_port", new_callable=AsyncMock) as mock_port, \
             patch("server._sbx_remove_sandbox", new_callable=AsyncMock) as mock_remove, \
             patch("server.os.killpg"), patch("server.os.getpgid", return_value=1):
            mock_run_command.return_value = (0, "", "")
            mock_port.return_value = None

            with self.assertRaises(RuntimeError):
                await server._spawn_session_server(Path("/tmp/p"), None, "reviewer")

        mock_remove.assert_awaited_once()

    async def test_deny_network_flag_added_for_none_profile(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])

        with patch("server.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("server._sbx_published_port", new_callable=AsyncMock, return_value=49152), \
             patch("server._get_agent_network_profile", return_value="none"):
            mock_run_command.return_value = (0, "", "")

            await server._spawn_session_server(Path("/tmp/p"), None, "reviewer")

        create_args = mock_run_command.await_args_list[0].args[0]
        self.assertIn("--deny-network", create_args)
        self.assertIn("**", create_args)

    async def test_passes_pane_env_to_exec(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])

        with patch("server.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as mock_exec, \
             patch("server._sbx_published_port", new_callable=AsyncMock, return_value=49152):
            mock_run_command.return_value = (0, "", "")

            await server._spawn_session_server(
                Path("/tmp/p"), ("term_1", "pane_1", "tab_1"), "reviewer"
            )

        exec_args = mock_exec.call_args.args
        self.assertIn("-e", exec_args)
        self.assertIn("ORCA_PANE_KEY=pane_1", exec_args)


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
        server._session_hook_identity.clear()

    @patch("server._attach_terminal_to_session", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server._spawn_session_server", new_callable=AsyncMock)
    @patch("server._create_session_pane", new_callable=AsyncMock)
    async def test_new_session_creates_pane_server_and_session(
        self, mock_create_pane, mock_spawn, mock_create_session, mock_attach
    ):
        process = FakeProcess([])
        mock_create_pane.return_value = ("term_1", "pane_1", "tab_1")
        mock_spawn.return_value = (process, "http://x", "orca-opencode-abc")
        mock_create_session.return_value = "ses_new"

        base_url, session_id = await server._create_session_backend(
            None, "reviewer", Path("/tmp/p")
        )

        self.assertEqual((base_url, session_id), ("http://x", "ses_new"))
        mock_spawn.assert_awaited_once_with(
            Path("/tmp/p"), ("term_1", "pane_1", "tab_1"), "reviewer"
        )
        mock_attach.assert_awaited_once_with("term_1", "http://x", "ses_new")
        self.assertEqual(
            server._session_state["ses_new"], ("orca-opencode-abc", "http://x", "term_1")
        )
        self.assertEqual(server._session_hook_identity["ses_new"], ("pane_1", "tab_1"))

    @patch("server._attach_terminal_to_session", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server._spawn_session_server", new_callable=AsyncMock)
    @patch("server._create_session_pane", new_callable=AsyncMock)
    async def test_falls_back_to_headless_when_orca_unavailable(
        self, mock_create_pane, mock_spawn, mock_create_session, mock_attach
    ):
        process = FakeProcess([])
        mock_create_pane.return_value = None
        mock_spawn.return_value = (process, "http://x", "orca-opencode-abc")
        mock_create_session.return_value = "ses_new"

        await server._create_session_backend(None, "reviewer", Path("/tmp/p"))

        mock_spawn.assert_awaited_once_with(Path("/tmp/p"), None, "reviewer")
        mock_attach.assert_not_awaited()
        self.assertEqual(
            server._session_state["ses_new"], ("orca-opencode-abc", "http://x", None)
        )
        self.assertNotIn("ses_new", server._session_hook_identity)

    @patch("server._attach_terminal_to_session", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server._spawn_session_server", new_callable=AsyncMock)
    @patch("server._create_session_pane", new_callable=AsyncMock)
    async def test_existing_session_id_skips_create_opencode_session(
        self, mock_create_pane, mock_spawn, mock_create_session, mock_attach
    ):
        process = FakeProcess([])
        mock_create_pane.return_value = None
        mock_spawn.return_value = (process, "http://x", "orca-opencode-abc")

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
    @patch("server._sbx_inspect", new_callable=AsyncMock)
    @patch("server._create_session_backend", new_callable=AsyncMock)
    async def test_reuses_when_server_and_terminal_alive(
        self, mock_create_backend, mock_inspect, mock_run_command
    ):
        server._session_state["ses_1"] = ("orca-opencode-abc", "http://x", "term_1")
        mock_inspect.return_value = {"name": "orca-opencode-abc", "status": "running"}
        mock_run_command.return_value = (0, "{}", "")

        result = await server.ensure_session_backend("ses_1", "reviewer", Path("/tmp/p"))

        self.assertEqual(result, ("http://x", "ses_1"))
        mock_create_backend.assert_not_awaited()

    @patch("server.run_command", new_callable=AsyncMock)
    @patch("server._sbx_inspect", new_callable=AsyncMock)
    @patch("server._create_session_backend", new_callable=AsyncMock)
    async def test_recreates_when_server_dead(
        self, mock_create_backend, mock_inspect, mock_run_command
    ):
        server._session_state["ses_1"] = ("orca-opencode-abc", "http://x", "term_1")
        mock_inspect.return_value = None
        mock_create_backend.return_value = ("http://y", "ses_1")

        result = await server.ensure_session_backend("ses_1", "reviewer", Path("/tmp/p"))

        self.assertEqual(result, ("http://y", "ses_1"))
        mock_create_backend.assert_awaited_once_with("ses_1", "reviewer", Path("/tmp/p"))
        mock_run_command.assert_not_awaited()  # server already dead, no need to check the terminal

    @patch("server.run_command", new_callable=AsyncMock)
    @patch("server._sbx_inspect", new_callable=AsyncMock)
    @patch("server._create_session_backend", new_callable=AsyncMock)
    async def test_recreates_when_terminal_gone(
        self, mock_create_backend, mock_inspect, mock_run_command
    ):
        server._session_state["ses_1"] = ("orca-opencode-abc", "http://x", "term_1")
        mock_inspect.return_value = {"name": "orca-opencode-abc", "status": "running"}
        mock_run_command.return_value = (1, "", "not found")
        mock_create_backend.return_value = ("http://y", "ses_1")

        result = await server.ensure_session_backend("ses_1", "reviewer", Path("/tmp/p"))

        self.assertEqual(result, ("http://y", "ses_1"))
        mock_create_backend.assert_awaited_once_with("ses_1", "reviewer", Path("/tmp/p"))

    @patch("server._sbx_inspect", new_callable=AsyncMock)
    @patch("server._create_session_backend", new_callable=AsyncMock)
    async def test_concurrent_calls_do_not_double_create(self, mock_create_backend, mock_inspect):
        call_count = 0
        mock_inspect.return_value = {"name": "orca-opencode-abc", "status": "running"}

        async def fake_create_backend(session_id, agent, project_dir):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            server._session_state[session_id] = ("orca-opencode-abc", "http://x", None)
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

    @patch("server.subprocess.run")
    def test_removes_every_tracked_sandbox(self, mock_run):
        server._session_state["ses_a"] = ("orca-opencode-a", "http://x", "term_a")
        server._session_state["ses_b"] = ("orca-opencode-b", "http://y", "term_b")

        server._kill_opencode_servers()

        called_names = {call.args[0][3] for call in mock_run.call_args_list}
        self.assertEqual(called_names, {"orca-opencode-a", "orca-opencode-b"})
        for call in mock_run.call_args_list:
            self.assertEqual(call.args[0][:2], ["sbx", "rm"])
            self.assertIn("--force", call.args[0])

    @patch("server.subprocess.run", side_effect=[Exception("boom"), None])
    def test_exception_on_one_does_not_block_others(self, mock_run):
        server._session_state["ses_c"] = ("orca-opencode-c", "http://x", None)
        server._session_state["ses_d"] = ("orca-opencode-d", "http://y", None)

        server._kill_opencode_servers()  # must not raise

        self.assertEqual(mock_run.call_count, 2)


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

    @patch("server._sbx_remove_sandbox", new_callable=AsyncMock)
    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_error_in_info_raises(
        self, mock_ensure_backend, mock_send_prompt, mock_remove_sandbox
    ):
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.return_value = {"info": {"error": {"message": "boom"}}, "parts": []}
        server._session_state["ses_1"] = ("orca-opencode-abc", "http://x", None)

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertIn("boom", str(ctx.exception))
        # An application-level error from OpenCode still means the run is
        # finished -- the sandbox must be torn down despite the exception.
        self.assertNotIn("ses_1", server._session_state)
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-abc")

    @patch("server._sbx_remove_sandbox", new_callable=AsyncMock)
    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_successful_run_evicts_and_kills_sandbox(
        self, mock_ensure_backend, mock_send_prompt, mock_remove_sandbox
    ):
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.return_value = {
            "info": {"finish": "stop"},
            "parts": [{"type": "text", "text": "ok"}],
        }
        server._session_state["ses_1"] = ("orca-opencode-abc", "http://x", None)

        await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertNotIn("ses_1", server._session_state)
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-abc")

    @patch("server._post_orca_status_hook", new_callable=AsyncMock)
    @patch("server._sbx_remove_sandbox", new_callable=AsyncMock)
    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_posts_busy_then_idle_status_hooks(
        self, mock_ensure_backend, mock_send_prompt, mock_remove_sandbox, mock_post_hook
    ):
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.return_value = {
            "info": {"finish": "stop"},
            "parts": [{"type": "text", "text": "ok"}],
        }

        await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(
            [call.args[:2] for call in mock_post_hook.await_args_list],
            [("ses_1", "SessionBusy"), ("ses_1", "SessionIdle")],
        )

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

    @patch("server._sbx_remove_sandbox", new_callable=AsyncMock)
    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_retries_once_after_transport_error(
        self, mock_ensure_backend, mock_send_prompt, mock_remove_sandbox
    ):
        mock_ensure_backend.side_effect = [
            ("http://first", "ses_new"),
            ("http://second", "ses_new"),
        ]
        mock_send_prompt.side_effect = [
            httpx2.ConnectError("boom"),
            {"info": {"finish": "stop"}, "parts": [{"type": "text", "text": "ok"}]},
        ]
        server._session_state["ses_new"] = ("orca-opencode-stale", "http://first", None)

        result = await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(result["response"], "ok")
        self.assertEqual(mock_ensure_backend.await_count, 2)
        self.assertEqual(mock_send_prompt.await_count, 2)
        self.assertNotIn("ses_new", server._session_state)
        # The stale sandbox must actually be removed, not just forgotten --
        # otherwise it leaks as an untracked, unkillable orphan container.
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-stale")
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

    @patch("server._post_orca_status_hook", new_callable=AsyncMock)
    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_session_backend", new_callable=AsyncMock)
    async def test_returns_permission_required_when_asked_before_prompt_completes(
        self, mock_ensure_backend, mock_send_prompt, mock_post_hook
    ):
        # Nobody is watching interactively: a pending "ask" must be surfaced
        # back to the caller instead of the call sitting until timeout.
        mock_ensure_backend.return_value = ("http://x", "ses_perm_1")

        async def hang_forever(*args, **kwargs):
            await asyncio.sleep(3600)

        mock_send_prompt.side_effect = hang_forever
        server._get_permission_queue("ses_perm_1").put_nowait(
            {"id": "perm_1", "permission": "bash", "patterns": ["rm *"]}
        )
        server._session_state["ses_perm_1"] = ("orca-opencode-abc", "http://x", None)

        try:
            result = await server.run_opencode(agent="tester", prompt="run it", session_id="ses_perm_1")

            self.assertEqual(result["status"], "permission_required")
            self.assertEqual(result["request_id"], "perm_1")
            self.assertEqual(result["action"], "bash")
            self.assertEqual(result["patterns"], ["rm *"])
            self.assertIn("ses_perm_1", server._pending_calls)
            # A pending permission must keep the sandbox alive for the
            # follow-up answer_permission() call.
            self.assertIn("ses_perm_1", server._session_state)
            self.assertEqual(
                [call.args[:2] for call in mock_post_hook.await_args_list],
                [("ses_perm_1", "SessionBusy"), ("ses_perm_1", "PermissionRequest")],
            )
        finally:
            server._pending_calls.pop("ses_perm_1").cancel()
            server._permission_queues.pop("ses_perm_1", None)
            server._session_state.pop("ses_perm_1", None)


class EvictAndKillSessionBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._session_state.clear()

    @patch("server._sbx_remove_sandbox", new_callable=AsyncMock)
    async def test_cancels_pending_call_and_clears_permission_queue(self, mock_remove_sandbox):
        server._session_state["ses_evict"] = ("orca-opencode-abc", "http://x", None)
        server._session_hook_identity["ses_evict"] = ("pane_1", "tab_1")

        async def hang_forever():
            await asyncio.sleep(3600)

        task = asyncio.ensure_future(hang_forever())
        server._pending_calls["ses_evict"] = task
        server._get_permission_queue("ses_evict")

        await server._evict_and_kill_session_backend("ses_evict")

        self.assertNotIn("ses_evict", server._pending_calls)
        self.assertNotIn("ses_evict", server._permission_queues)
        self.assertNotIn("ses_evict", server._session_hook_identity)
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-abc")
        with self.assertRaises(asyncio.CancelledError):
            await task


class AnswerPermissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._session_state.clear()

    def tearDown(self):
        server._session_state.pop("ses_ans", None)
        server._pending_calls.pop("ses_ans", None)
        server._permission_queues.pop("ses_ans", None)

    async def test_unknown_session_raises(self):
        with self.assertRaises(server.ToolError):
            await server.answer_permission(session_id="nope", request_id="r1", reply="once")

    async def test_invalid_reply_raises(self):
        server._session_state["ses_ans"] = (None, "http://x", None)

        with self.assertRaises(server.ToolError):
            await server.answer_permission(session_id="ses_ans", request_id="r1", reply="bogus")

    async def test_posts_reply_and_returns_ok_when_nothing_pending(self):
        server._session_state["ses_ans"] = (None, "http://x", None)
        mock_client = make_mock_client(FakeResponse(json_data={}))

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            result = await server.answer_permission(
                session_id="ses_ans", request_id="perm_1", reply="once"
            )

        self.assertEqual(result, {"session_id": "ses_ans", "status": "ok"})
        mock_client.post.assert_awaited_once_with(
            "http://x/session/ses_ans/permissions/perm_1",
            json={"response": "once"},
        )

    async def test_reply_error_status_raises(self):
        server._session_state["ses_ans"] = (None, "http://x", None)
        mock_client = make_mock_client(FakeResponse(status_code=500, text="boom"))

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(server.ToolError) as ctx:
                await server.answer_permission(
                    session_id="ses_ans", request_id="perm_1", reply="once"
                )

        self.assertIn("boom", str(ctx.exception))

    @patch("server._post_orca_status_hook", new_callable=AsyncMock)
    @patch("server._sbx_remove_sandbox", new_callable=AsyncMock)
    async def test_resumes_pending_call_and_returns_final_result(
        self, mock_remove_sandbox, mock_post_hook
    ):
        server._session_state["ses_ans"] = ("orca-opencode-abc", "http://x", None)
        mock_client = make_mock_client(FakeResponse(json_data={}))

        async def eventually_done():
            return {"info": {"finish": "stop"}, "parts": [{"type": "text", "text": "ok"}]}

        server._pending_calls["ses_ans"] = asyncio.ensure_future(eventually_done())

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            result = await server.answer_permission(
                session_id="ses_ans", request_id="perm_1", reply="once"
            )

        self.assertEqual(
            result, {"session_id": "ses_ans", "response": "ok", "finish_reason": "stop"}
        )
        self.assertNotIn("ses_ans", server._pending_calls)
        # A finished run must tear its sandbox down instead of leaving it
        # running until the whole MCP process exits.
        self.assertNotIn("ses_ans", server._session_state)
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-abc")
        self.assertEqual(
            [call.args[:2] for call in mock_post_hook.await_args_list],
            [("ses_ans", "SessionBusy"), ("ses_ans", "SessionIdle")],
        )

    @patch("server._post_orca_status_hook", new_callable=AsyncMock)
    async def test_resumed_call_hits_another_permission_request(self, mock_post_hook):
        server._session_state["ses_ans"] = (None, "http://x", None)
        mock_client = make_mock_client(FakeResponse(json_data={}))

        async def hang_forever():
            await asyncio.sleep(3600)

        server._pending_calls["ses_ans"] = asyncio.ensure_future(hang_forever())
        server._get_permission_queue("ses_ans").put_nowait(
            {"id": "perm_2", "permission": "edit", "patterns": ["*.env"]}
        )

        with patch("server.httpx2.AsyncClient", return_value=mock_client):
            result = await server.answer_permission(
                session_id="ses_ans", request_id="perm_1", reply="once"
            )

        self.assertEqual(result["status"], "permission_required")
        self.assertEqual(result["request_id"], "perm_2")
        # A chained permission request must not tear the sandbox down --
        # the session is still needed for the next answer_permission call.
        self.assertIn("ses_ans", server._session_state)
        self.assertEqual(
            [call.args[:2] for call in mock_post_hook.await_args_list],
            [("ses_ans", "SessionBusy"), ("ses_ans", "PermissionRequest")],
        )
        server._pending_calls.pop("ses_ans").cancel()


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
