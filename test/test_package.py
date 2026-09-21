import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import package


class ParseJsonOutputTests(unittest.TestCase):
    def test_single_line_json(self):
        output = '{"ok": true, "value": 1}'
        self.assertEqual(package.parse_json_output(output), {"ok": True, "value": 1})

    def test_pretty_printed_multiline_json(self):
        output = """{
  "ok": true,
  "result": {
    "terminal": {
      "handle": "term_abc"
    }
  }
}"""
        result = package.parse_json_output(output)
        self.assertEqual(result["result"]["terminal"]["handle"], "term_abc")

    def test_fallback_finds_json_after_leading_banner_line(self):
        output = "some warning printed to stdout\n{\"ok\": true}"
        self.assertEqual(package.parse_json_output(output), {"ok": True})

    def test_fallback_picks_first_valid_object_when_two_are_present(self):
        # Documents the actual invariant: first-valid-dict-wins, not last.
        output = '{"first": 1}\n{"second": 2}'
        self.assertEqual(package.parse_json_output(output), {"first": 1})

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
        result = package.parse_json_output(output)
        self.assertEqual(result["result"]["terminal"]["handle"], "term_abc")

    def test_pretty_printed_json_with_trailing_banner(self):
        output = """{
  "ok": true,
  "value": 42
}
trailing log noise after the JSON"""
        self.assertEqual(package.parse_json_output(output), {"ok": True, "value": 42})

    def test_unparsable_output_raises(self):
        with self.assertRaises(RuntimeError):
            package.parse_json_output("not json at all")

    def test_empty_output_raises(self):
        with self.assertRaises(RuntimeError):
            package.parse_json_output("")

    def test_non_dict_json_raises(self):
        with self.assertRaises(RuntimeError):
            package.parse_json_output("[1, 2, 3]")


class RunCommandTests(unittest.IsolatedAsyncioTestCase):
    """Exercises real subprocesses (no mocking) since this is the foundation
    every sandbox/integration call goes through."""

    async def test_success_returns_code_and_output(self):
        returncode, stdout, stderr = await package.run_command(["echo", "hello"])
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout.strip(), "hello")
        self.assertEqual(stderr, "")

    async def test_nonzero_return_code(self):
        returncode, _, _ = await package.run_command(["sh", "-c", "exit 3"])
        self.assertEqual(returncode, 3)

    async def test_stderr_is_captured(self):
        _, _, stderr = await package.run_command(["sh", "-c", "echo oops 1>&2"])
        self.assertEqual(stderr.strip(), "oops")

    async def test_cwd_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, stdout, _ = await package.run_command(["pwd"], cwd=tmp_dir)
            self.assertEqual(
                os.path.realpath(stdout.strip()), os.path.realpath(tmp_dir)
            )

    async def test_timeout_raises_and_kills_process(self):
        with self.assertRaises(RuntimeError) as ctx:
            await package.run_command(["sleep", "5"], timeout=0.2)

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

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            session_id = await package.create_opencode_session("http://127.0.0.1:4096", title="t")

        self.assertEqual(session_id, "ses_123")
        mock_client.post.assert_awaited_once_with(
            "http://127.0.0.1:4096/session", json={"title": "t"}
        )

    async def test_error_status_raises(self):
        mock_client = make_mock_client(FakeResponse(status_code=400, text="bad request"))

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await package.create_opencode_session("http://x", title="t")

        self.assertIn("bad request", str(ctx.exception))

    async def test_missing_id_raises(self):
        mock_client = make_mock_client(FakeResponse(json_data={}))

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError):
                await package.create_opencode_session("http://x", title="t")

    async def test_non_dict_response_raises(self):
        mock_client = make_mock_client(FakeResponse(json_data=["not", "a", "dict"]))

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError):
                await package.create_opencode_session("http://x", title="t")


class SendOpencodePromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_expected_body_and_returns_json(self):
        payload = {"info": {"finish": "stop"}, "parts": [{"type": "text", "text": "hi"}]}
        mock_client = make_mock_client(FakeResponse(json_data=payload))

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            result = await package.send_opencode_prompt(
                "http://x", "ses_1", "reviewer", "do it", 30
            )

        self.assertEqual(result, payload)
        mock_client.post.assert_awaited_once_with(
            "http://x/session/ses_1/message",
            json={"agent": "reviewer", "parts": [{"type": "text", "text": "do it"}]},
        )

    async def test_error_status_raises_with_body(self):
        mock_client = make_mock_client(FakeResponse(status_code=500, text="boom"))

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await package.send_opencode_prompt("http://x", "ses_1", "reviewer", "do it", 30)

        self.assertIn("boom", str(ctx.exception))

    async def test_non_dict_response_raises(self):
        mock_client = make_mock_client(FakeResponse(json_data=["not", "a", "dict"]))

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError):
                await package.send_opencode_prompt("http://x", "ses_1", "reviewer", "do it", 30)


# ---------------------------------------------------------------------------
# Per-session opencode serve process management.
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


class RecordingVisibilityHook(package.VisibilityHook):
    """A no-op-by-default VisibilityHook that records calls and lets tests
    control extra_env()/is_alive() -- stands in for a real integration."""

    def __init__(self, before_spawn_result=None, env=None, is_alive_result=True):
        self.before_spawn_result = before_spawn_result
        self.env = dict(env or {})
        self.is_alive_result = is_alive_result
        self.after_spawn_calls: list[tuple] = []
        self.is_alive_calls = 0

    async def before_spawn(self, agent, project_dir):
        return self.before_spawn_result

    async def after_spawn(self, handle, base_url, session_id):
        self.after_spawn_calls.append((handle, base_url, session_id))

    async def is_alive(self, handle):
        self.is_alive_calls += 1
        return self.is_alive_result

    def extra_env(self, handle):
        return dict(self.env)


class RecordingStatusHook(package.StatusHook):
    def __init__(self):
        self.events: list[tuple] = []

    async def post_status(self, session_id, event_name, properties=None):
        self.events.append((session_id, event_name, properties))


class SandboxExecEnvTests(unittest.TestCase):
    def test_merges_hook_extra_env(self):
        hook = RecordingVisibilityHook(env={"CUSTOM": "1"})

        env = package._sandbox_exec_env("handle", hook)

        self.assertEqual(env.get("CUSTOM"), "1")

    def test_no_extra_env_when_hook_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            env = package._sandbox_exec_env(None, package.VisibilityHook())

        self.assertEqual(env, {})

    def test_forwards_hooks_dir_regardless_of_hook_env(self):
        with patch.dict(os.environ, {"OPENCODE_CONFIG_DIR": "/x"}):
            env = package._sandbox_exec_env(None, package.VisibilityHook())

        self.assertEqual(env.get("OPENCODE_CONFIG_DIR"), "/x")


class GetAgentNetworkProfileTests(unittest.TestCase):
    def test_defaults_to_balanced_when_no_definition_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = package._get_agent_network_profile("nope", Path(tmpdir))

        self.assertEqual(profile, "balanced")

    def test_reads_declared_profile_from_project_local_definition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            agent_dir = project_dir / ".opencode" / "agent"
            agent_dir.mkdir(parents=True)
            (agent_dir / "reviewer.md").write_text(
                "---\ndescription: x\nnetwork: none\n---\nbody", encoding="utf-8"
            )

            profile = package._get_agent_network_profile("reviewer", project_dir)

        self.assertEqual(profile, "none")

    def test_invalid_value_falls_back_to_balanced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            agent_dir = project_dir / ".opencode" / "agent"
            agent_dir.mkdir(parents=True)
            (agent_dir / "reviewer.md").write_text(
                "---\ndescription: x\nnetwork: wide-open\n---\nbody", encoding="utf-8"
            )

            profile = package._get_agent_network_profile("reviewer", project_dir)

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
                "core.package.asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = process
                await package._inject_agent_definition("sbx_1", "reviewer", source_path)

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
                "core.package.asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = process
                await package._inject_agent_definition("sbx_1", "reviewer", source_path)  # must not raise

    async def test_exception_is_swallowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "reviewer.md"
            source_path.write_text("body", encoding="utf-8")

            with patch(
                "core.package.asyncio.create_subprocess_exec", side_effect=OSError("boom")
            ):
                await package._inject_agent_definition("sbx_1", "reviewer", source_path)  # must not raise


class SbxInspectTests(unittest.IsolatedAsyncioTestCase):
    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_finds_matching_sandbox(self, mock_run_command):
        mock_run_command.return_value = (
            0,
            json.dumps({"sandboxes": [{"name": "orca-opencode-abc", "status": "running"}]}),
            "",
        )

        sandbox = await package._sbx_inspect("orca-opencode-abc")

        self.assertEqual(sandbox["status"], "running")

    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_returns_none_when_not_found(self, mock_run_command):
        mock_run_command.return_value = (0, json.dumps({"sandboxes": []}), "")

        self.assertIsNone(await package._sbx_inspect("missing"))

    @patch("core.package.run_command", new_callable=AsyncMock)
    async def test_raises_on_command_failure(self, mock_run_command):
        mock_run_command.return_value = (1, "", "daemon not running")

        with self.assertRaises(RuntimeError):
            await package._sbx_inspect("any")

    @patch("core.package._sbx_inspect", new_callable=AsyncMock)
    async def test_published_port_matches_by_sandbox_port(self, mock_inspect):
        mock_inspect.return_value = {
            "ports": [{"sandbox_port": 4096, "host_port": 49152}]
        }

        port = await package._sbx_published_port("orca-opencode-abc", 4096)

        self.assertEqual(port, 49152)

    @patch("core.package._sbx_inspect", new_callable=AsyncMock)
    async def test_published_port_none_when_sandbox_missing(self, mock_inspect):
        mock_inspect.return_value = None

        self.assertIsNone(await package._sbx_published_port("missing", 4096))


class SpawnSessionServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch("core.package._find_agent_definition", return_value=None)
        self.addCleanup(patcher.stop)
        patcher.start()
        patcher = patch("core.package._inject_agent_definition", new_callable=AsyncMock)
        self.addCleanup(patcher.stop)
        patcher.start()
        patcher = patch("core.package._get_agent_network_profile", return_value="balanced")
        self.addCleanup(patcher.stop)
        patcher.start()

    async def test_injects_global_agent_definition_before_starting_serve(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])
        global_path = Path("/home/host-user/.config/opencode/agent/reviewer.md")

        with patch("core.package.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("core.package.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("core.package._sbx_published_port", new_callable=AsyncMock, return_value=49152), \
             patch("core.package._find_agent_definition", return_value=global_path), \
             patch("core.package._inject_agent_definition", new_callable=AsyncMock) as mock_inject:
            mock_run_command.return_value = (0, "", "")

            _process, _base_url, sandbox_name = await package._spawn_session_server(
                Path("/tmp/p"), None, "reviewer", package.VisibilityHook()
            )

        mock_inject.assert_awaited_once_with(sandbox_name, "reviewer", global_path)

    async def test_skips_injection_for_project_local_definition(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])
        project_local_path = Path("/tmp/p/.opencode/agent/reviewer.md")

        with patch("core.package.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("core.package.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("core.package._sbx_published_port", new_callable=AsyncMock, return_value=49152), \
             patch("core.package._find_agent_definition", return_value=project_local_path), \
             patch("core.package._inject_agent_definition", new_callable=AsyncMock) as mock_inject:
            mock_run_command.return_value = (0, "", "")

            await package._spawn_session_server(Path("/tmp/p"), None, "reviewer", package.VisibilityHook())

        mock_inject.assert_not_awaited()

    async def test_returns_process_base_url_and_sandbox_name(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])

        with patch("core.package.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("core.package.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("core.package._sbx_published_port", new_callable=AsyncMock) as mock_port:
            mock_run_command.return_value = (0, "", "")
            mock_port.return_value = 49152

            result_process, base_url, sandbox_name = await package._spawn_session_server(
                Path("/tmp/p"), None, "reviewer", package.VisibilityHook()
            )

        self.assertIs(result_process, process)
        self.assertEqual(base_url, "http://127.0.0.1:49152")
        self.assertTrue(sandbox_name.startswith("opencode-bridge-"))

    async def test_sbx_create_failure_raises(self):
        with patch("core.package.run_command", new_callable=AsyncMock) as mock_run_command:
            mock_run_command.return_value = (1, "", "sbx: image pull failed")

            with self.assertRaises(RuntimeError):
                await package._spawn_session_server(
                    Path("/tmp/p"), None, "reviewer", package.VisibilityHook()
                )

    @patch("core.package.os.killpg")
    @patch("core.package.os.getpgid", return_value=1)
    async def test_process_exits_without_listening_line_raises_and_removes_sandbox(
        self, mock_getpgid, mock_killpg
    ):
        process = FakeProcess([b"some startup noise\n"], returncode=1)

        with patch("core.package.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("core.package.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("core.package._sbx_remove_sandbox", new_callable=AsyncMock) as mock_remove:
            mock_run_command.return_value = (0, "", "")

            with self.assertRaises(RuntimeError):
                await package._spawn_session_server(
                    Path("/tmp/p"), None, "reviewer", package.VisibilityHook()
                )

        mock_remove.assert_awaited_once()

    @patch("core.package.os.killpg")
    @patch("core.package.os.getpgid", return_value=1)
    @patch("core.package.OPENCODE_SERVE_STARTUP_TIMEOUT_SECONDS", 0.05)
    async def test_timeout_raises_kills_process_and_removes_sandbox(
        self, mock_getpgid, mock_killpg
    ):
        process = FakeProcess(hang=True)

        with patch("core.package.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("core.package.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("core.package._sbx_remove_sandbox", new_callable=AsyncMock) as mock_remove:
            mock_run_command.return_value = (0, "", "")

            with self.assertRaises(asyncio.TimeoutError):
                await package._spawn_session_server(
                    Path("/tmp/p"), None, "reviewer", package.VisibilityHook()
                )

        mock_killpg.assert_called_once()
        mock_remove.assert_awaited_once()

    async def test_published_port_missing_raises_and_removes_sandbox(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])

        with patch("core.package.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("core.package.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("core.package._sbx_published_port", new_callable=AsyncMock) as mock_port, \
             patch("core.package._sbx_remove_sandbox", new_callable=AsyncMock) as mock_remove, \
             patch("core.package.os.killpg"), patch("core.package.os.getpgid", return_value=1):
            mock_run_command.return_value = (0, "", "")
            mock_port.return_value = None

            with self.assertRaises(RuntimeError):
                await package._spawn_session_server(
                    Path("/tmp/p"), None, "reviewer", package.VisibilityHook()
                )

        mock_remove.assert_awaited_once()

    async def test_deny_network_flag_added_for_none_profile(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])

        with patch("core.package.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("core.package.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)), \
             patch("core.package._sbx_published_port", new_callable=AsyncMock, return_value=49152), \
             patch("core.package._get_agent_network_profile", return_value="none"):
            mock_run_command.return_value = (0, "", "")

            await package._spawn_session_server(
                Path("/tmp/p"), None, "reviewer", package.VisibilityHook()
            )

        create_args = mock_run_command.await_args_list[0].args[0]
        self.assertIn("--deny-network", create_args)
        self.assertIn("**", create_args)

    async def test_passes_hook_env_to_exec(self):
        process = FakeProcess([b"opencode server listening on http://0.0.0.0:4096\n"])
        hook = RecordingVisibilityHook(env={"CUSTOM_ENV": "value1"})

        with patch("core.package.run_command", new_callable=AsyncMock) as mock_run_command, \
             patch("core.package.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as mock_exec, \
             patch("core.package._sbx_published_port", new_callable=AsyncMock, return_value=49152):
            mock_run_command.return_value = (0, "", "")

            await package._spawn_session_server(Path("/tmp/p"), "handle_1", "reviewer", hook)

        exec_args = mock_exec.call_args.args
        self.assertIn("-e", exec_args)
        self.assertIn("CUSTOM_ENV=value1", exec_args)


class KillAllSandboxesTests(unittest.TestCase):
    def setUp(self):
        package._session_state.clear()

    @patch("core.package.subprocess.run")
    def test_removes_every_tracked_sandbox(self, mock_run):
        package._session_state["ses_a"] = ("orca-opencode-a", "http://x", "handle_a")
        package._session_state["ses_b"] = ("orca-opencode-b", "http://y", "handle_b")

        package._kill_all_sandboxes()

        called_names = {call.args[0][3] for call in mock_run.call_args_list}
        self.assertEqual(called_names, {"orca-opencode-a", "orca-opencode-b"})
        for call in mock_run.call_args_list:
            self.assertEqual(call.args[0][:2], ["sbx", "rm"])
            self.assertIn("--force", call.args[0])

    @patch("core.package.subprocess.run", side_effect=[Exception("boom"), None])
    def test_exception_on_one_does_not_block_others(self, mock_run):
        package._session_state["ses_c"] = ("orca-opencode-c", "http://x", None)
        package._session_state["ses_d"] = ("orca-opencode-d", "http://y", None)

        package._kill_all_sandboxes()  # must not raise

        self.assertEqual(mock_run.call_count, 2)


class InstallTerminationCleanupTests(unittest.TestCase):
    def test_chains_to_previous_callable_handler(self):
        previous = MagicMock()
        installed_handlers = {}

        with patch("core.package.signal.getsignal", return_value=previous), \
             patch("core.package.signal.signal", side_effect=lambda sig, h: installed_handlers.__setitem__(sig, h)), \
             patch("core.package._kill_all_sandboxes") as mock_kill:
            package._install_termination_cleanup()

            self.assertIn(package.signal.SIGTERM, installed_handlers)
            installed_handlers[package.signal.SIGTERM](package.signal.SIGTERM, None)

            mock_kill.assert_called_once()
        previous.assert_called_once_with(package.signal.SIGTERM, None)

    def test_falls_back_to_default_when_previous_is_sig_dfl(self):
        installed_handlers = {}

        with patch("core.package.signal.getsignal", return_value=package.signal.SIG_DFL), \
             patch("core.package.signal.signal", side_effect=lambda sig, h: installed_handlers.__setitem__(sig, h)), \
             patch("core.package._kill_all_sandboxes") as mock_kill, \
             patch("core.package.os.kill") as mock_os_kill:
            package._install_termination_cleanup()
            installed_handlers[package.signal.SIGTERM](package.signal.SIGTERM, None)

            mock_kill.assert_called_once()
            mock_os_kill.assert_called_once()

    def test_leaves_sig_ign_untouched(self):
        installed_handlers = {}

        with patch("core.package.signal.getsignal", return_value=package.signal.SIG_IGN), \
             patch("core.package.signal.signal", side_effect=lambda sig, h: installed_handlers.__setitem__(sig, h)), \
             patch("core.package._kill_all_sandboxes") as mock_kill, \
             patch("core.package.os.kill") as mock_os_kill:
            package._install_termination_cleanup()
            installed_handlers[package.signal.SIGTERM](package.signal.SIGTERM, None)

            mock_kill.assert_called_once()
            mock_os_kill.assert_not_called()


class CreateSessionBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        package._session_state.clear()

    @patch("core.package.create_opencode_session", new_callable=AsyncMock)
    @patch("core.package._spawn_session_server", new_callable=AsyncMock)
    async def test_new_session_creates_backend_via_hooks(self, mock_spawn, mock_create_session):
        process = FakeProcess([])
        hook = RecordingVisibilityHook(before_spawn_result="handle_1")
        mock_spawn.return_value = (process, "http://x", "orca-opencode-abc")
        mock_create_session.return_value = "ses_new"

        base_url, session_id = await package._create_session_backend(
            None, "reviewer", Path("/tmp/p"), hook
        )

        self.assertEqual((base_url, session_id), ("http://x", "ses_new"))
        mock_spawn.assert_awaited_once_with(Path("/tmp/p"), "handle_1", "reviewer", hook)
        self.assertEqual(hook.after_spawn_calls, [("handle_1", "http://x", "ses_new")])
        self.assertEqual(
            package._session_state["ses_new"], ("orca-opencode-abc", "http://x", "handle_1")
        )

    @patch("core.package.create_opencode_session", new_callable=AsyncMock)
    @patch("core.package._spawn_session_server", new_callable=AsyncMock)
    async def test_handles_hook_returning_none(self, mock_spawn, mock_create_session):
        process = FakeProcess([])
        hook = RecordingVisibilityHook(before_spawn_result=None)
        mock_spawn.return_value = (process, "http://x", "orca-opencode-abc")
        mock_create_session.return_value = "ses_new"

        await package._create_session_backend(None, "reviewer", Path("/tmp/p"), hook)

        mock_spawn.assert_awaited_once_with(Path("/tmp/p"), None, "reviewer", hook)
        self.assertEqual(hook.after_spawn_calls, [(None, "http://x", "ses_new")])
        self.assertEqual(
            package._session_state["ses_new"], ("orca-opencode-abc", "http://x", None)
        )

    @patch("core.package.create_opencode_session", new_callable=AsyncMock)
    @patch("core.package._spawn_session_server", new_callable=AsyncMock)
    async def test_existing_session_id_skips_create_opencode_session(
        self, mock_spawn, mock_create_session
    ):
        process = FakeProcess([])
        hook = RecordingVisibilityHook(before_spawn_result=None)
        mock_spawn.return_value = (process, "http://x", "orca-opencode-abc")

        base_url, session_id = await package._create_session_backend(
            "ses_existing", "reviewer", Path("/tmp/p"), hook
        )

        self.assertEqual(session_id, "ses_existing")
        mock_create_session.assert_not_awaited()


class EnsureSessionBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        package._session_state.clear()

    @patch("core.package._create_session_backend", new_callable=AsyncMock)
    async def test_new_session_always_creates(self, mock_create_backend):
        mock_create_backend.return_value = ("http://x", "ses_new")
        hook = package.VisibilityHook()

        result = await package.ensure_session_backend(None, "reviewer", Path("/tmp/p"), hook)

        self.assertEqual(result, ("http://x", "ses_new"))
        mock_create_backend.assert_awaited_once_with(None, "reviewer", Path("/tmp/p"), hook)

    @patch("core.package._sbx_inspect", new_callable=AsyncMock)
    @patch("core.package._create_session_backend", new_callable=AsyncMock)
    async def test_reuses_when_server_and_handle_alive(self, mock_create_backend, mock_inspect):
        package._session_state["ses_1"] = ("orca-opencode-abc", "http://x", "handle_1")
        mock_inspect.return_value = {"name": "orca-opencode-abc", "status": "running"}
        hook = RecordingVisibilityHook(is_alive_result=True)

        result = await package.ensure_session_backend("ses_1", "reviewer", Path("/tmp/p"), hook)

        self.assertEqual(result, ("http://x", "ses_1"))
        mock_create_backend.assert_not_awaited()
        self.assertEqual(hook.is_alive_calls, 1)

    @patch("core.package._sbx_inspect", new_callable=AsyncMock)
    @patch("core.package._create_session_backend", new_callable=AsyncMock)
    async def test_recreates_when_server_dead(self, mock_create_backend, mock_inspect):
        package._session_state["ses_1"] = ("orca-opencode-abc", "http://x", "handle_1")
        mock_inspect.return_value = None
        mock_create_backend.return_value = ("http://y", "ses_1")
        hook = RecordingVisibilityHook()

        result = await package.ensure_session_backend("ses_1", "reviewer", Path("/tmp/p"), hook)

        self.assertEqual(result, ("http://y", "ses_1"))
        mock_create_backend.assert_awaited_once_with("ses_1", "reviewer", Path("/tmp/p"), hook)
        self.assertEqual(hook.is_alive_calls, 0)  # server already dead, no need to check the handle

    @patch("core.package._sbx_inspect", new_callable=AsyncMock)
    @patch("core.package._create_session_backend", new_callable=AsyncMock)
    async def test_recreates_when_handle_dead(self, mock_create_backend, mock_inspect):
        package._session_state["ses_1"] = ("orca-opencode-abc", "http://x", "handle_1")
        mock_inspect.return_value = {"name": "orca-opencode-abc", "status": "running"}
        mock_create_backend.return_value = ("http://y", "ses_1")
        hook = RecordingVisibilityHook(is_alive_result=False)

        result = await package.ensure_session_backend("ses_1", "reviewer", Path("/tmp/p"), hook)

        self.assertEqual(result, ("http://y", "ses_1"))
        mock_create_backend.assert_awaited_once_with("ses_1", "reviewer", Path("/tmp/p"), hook)

    @patch("core.package._sbx_inspect", new_callable=AsyncMock)
    @patch("core.package._create_session_backend", new_callable=AsyncMock)
    async def test_concurrent_calls_do_not_double_create(self, mock_create_backend, mock_inspect):
        call_count = 0
        mock_inspect.return_value = {"name": "orca-opencode-abc", "status": "running"}
        hook = RecordingVisibilityHook()

        async def fake_create_backend(session_id, agent, project_dir, visibility_hook):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            package._session_state[session_id] = ("orca-opencode-abc", "http://x", None)
            return "http://x", session_id

        mock_create_backend.side_effect = fake_create_backend

        await asyncio.gather(
            package.ensure_session_backend("ses_concurrent", "reviewer", Path("/tmp/p"), hook),
            package.ensure_session_backend("ses_concurrent", "reviewer", Path("/tmp/p"), hook),
        )

        self.assertEqual(call_count, 1)


# ---------------------------------------------------------------------------
# run_opencode orchestration.
# ---------------------------------------------------------------------------


class RunOpencodeTests(unittest.IsolatedAsyncioTestCase):
    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
    async def test_creates_new_session_when_none_given(
        self, mock_ensure_backend, mock_send_prompt
    ):
        hook = package.VisibilityHook()
        mock_ensure_backend.return_value = ("http://127.0.0.1:4096", "ses_new")
        mock_send_prompt.return_value = {
            "info": {"finish": "stop"},
            "parts": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}],
        }

        result = await package.run_opencode(agent="reviewer", prompt="do it", visibility_hook=hook)

        self.assertEqual(
            result, {"session_id": "ses_new", "response": "hello world", "finish_reason": "stop"}
        )
        mock_ensure_backend.assert_awaited_once_with(None, "reviewer", package.get_project_dir(), hook)
        mock_send_prompt.assert_awaited_once_with(
            "http://127.0.0.1:4096", "ses_new", "reviewer", "do it",
            package.OPENCODE_PROMPT_TIMEOUT_SECONDS,
        )

    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
    async def test_reuses_given_session_id(self, mock_ensure_backend, mock_send_prompt):
        hook = package.VisibilityHook()
        mock_ensure_backend.return_value = ("http://x", "ses_existing")
        mock_send_prompt.return_value = {"info": {"finish": "stop"}, "parts": []}

        result = await package.run_opencode(
            agent="reviewer", prompt="do it", session_id="ses_existing", visibility_hook=hook
        )

        self.assertEqual(result["session_id"], "ses_existing")
        mock_ensure_backend.assert_awaited_once_with(
            "ses_existing", "reviewer", package.get_project_dir(), hook
        )

    @patch("core.package._sbx_remove_sandbox", new_callable=AsyncMock)
    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
    async def test_error_in_info_raises(
        self, mock_ensure_backend, mock_send_prompt, mock_remove_sandbox
    ):
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.return_value = {"info": {"error": {"message": "boom"}}, "parts": []}
        package._session_state["ses_1"] = ("orca-opencode-abc", "http://x", None)

        with self.assertRaises(RuntimeError) as ctx:
            await package.run_opencode(agent="reviewer", prompt="do it")

        self.assertIn("boom", str(ctx.exception))
        # An application-level error from OpenCode still means the run is
        # finished -- the sandbox must be torn down despite the exception.
        self.assertNotIn("ses_1", package._session_state)
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-abc")

    @patch("core.package._sbx_remove_sandbox", new_callable=AsyncMock)
    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
    async def test_successful_run_evicts_and_kills_sandbox(
        self, mock_ensure_backend, mock_send_prompt, mock_remove_sandbox
    ):
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.return_value = {
            "info": {"finish": "stop"},
            "parts": [{"type": "text", "text": "ok"}],
        }
        package._session_state["ses_1"] = ("orca-opencode-abc", "http://x", None)

        await package.run_opencode(agent="reviewer", prompt="do it")

        self.assertNotIn("ses_1", package._session_state)
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-abc")

    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
    async def test_posts_busy_then_idle_status_hooks(
        self, mock_ensure_backend, mock_send_prompt
    ):
        status_hook = RecordingStatusHook()
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.return_value = {
            "info": {"finish": "stop"},
            "parts": [{"type": "text", "text": "ok"}],
        }

        await package.run_opencode(agent="reviewer", prompt="do it", status_hook=status_hook)

        self.assertEqual(
            [event[:2] for event in status_hook.events],
            [("ses_1", "SessionBusy"), ("ses_1", "SessionIdle")],
        )

    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
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

        result = await package.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(result["response"], "kept")

    @patch("core.package._sbx_remove_sandbox", new_callable=AsyncMock)
    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
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
        package._session_state["ses_new"] = ("orca-opencode-stale", "http://first", None)

        result = await package.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(result["response"], "ok")
        self.assertEqual(mock_ensure_backend.await_count, 2)
        self.assertEqual(mock_send_prompt.await_count, 2)
        self.assertNotIn("ses_new", package._session_state)
        # The stale sandbox must actually be removed, not just forgotten --
        # otherwise it leaks as an untracked, unkillable orphan container.
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-stale")
        # The second attempt must reuse the session_id learned from the first
        # (not start over with session_id=None), so history isn't lost.
        second_call_args = mock_ensure_backend.await_args_list[1].args
        self.assertEqual(second_call_args[0], "ses_new")

    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
    async def test_read_timeout_does_not_trigger_retry(
        self, mock_ensure_backend, mock_send_prompt
    ):
        # A slow-but-connected call must not be torn down and retried just
        # because it took a while -- only actual connection failures should.
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.side_effect = httpx2.ReadTimeout("slow")

        with self.assertRaises(httpx2.ReadTimeout):
            await package.run_opencode(agent="reviewer", prompt="do it")

        mock_send_prompt.assert_awaited_once()

    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
    async def test_raises_after_second_transport_error(
        self, mock_ensure_backend, mock_send_prompt
    ):
        mock_ensure_backend.return_value = ("http://x", "ses_1")
        mock_send_prompt.side_effect = httpx2.ConnectError("boom")

        with self.assertRaises(httpx2.TransportError):
            await package.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(mock_send_prompt.await_count, 2)

    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
    async def test_retry_reuses_explicit_session_id(
        self, mock_ensure_backend, mock_send_prompt
    ):
        mock_ensure_backend.return_value = ("http://x", "ses_existing")
        mock_send_prompt.side_effect = [
            httpx2.ConnectError("boom"),
            {"info": {"finish": "stop"}, "parts": []},
        ]

        result = await package.run_opencode(agent="reviewer", prompt="do it", session_id="ses_existing")

        self.assertEqual(result["session_id"], "ses_existing")
        for call in mock_ensure_backend.await_args_list:
            self.assertEqual(call.args[0], "ses_existing")

    @patch("core.package.send_opencode_prompt", new_callable=AsyncMock)
    @patch("core.package.ensure_session_backend", new_callable=AsyncMock)
    async def test_returns_permission_required_when_asked_before_prompt_completes(
        self, mock_ensure_backend, mock_send_prompt
    ):
        # Nobody is watching interactively: a pending "ask" must be surfaced
        # back to the caller instead of the call sitting until timeout.
        status_hook = RecordingStatusHook()
        mock_ensure_backend.return_value = ("http://x", "ses_perm_1")

        async def hang_forever(*args, **kwargs):
            await asyncio.sleep(3600)

        mock_send_prompt.side_effect = hang_forever
        package._get_permission_queue("ses_perm_1").put_nowait(
            {"id": "perm_1", "permission": "bash", "patterns": ["rm *"]}
        )
        package._session_state["ses_perm_1"] = ("orca-opencode-abc", "http://x", None)

        try:
            result = await package.run_opencode(
                agent="tester", prompt="run it", session_id="ses_perm_1", status_hook=status_hook
            )

            self.assertEqual(result["status"], "permission_required")
            self.assertEqual(result["request_id"], "perm_1")
            self.assertEqual(result["action"], "bash")
            self.assertEqual(result["patterns"], ["rm *"])
            self.assertIn("ses_perm_1", package._pending_calls)
            # A pending permission must keep the sandbox alive for the
            # follow-up answer_permission() call.
            self.assertIn("ses_perm_1", package._session_state)
            self.assertEqual(
                [event[:2] for event in status_hook.events],
                [("ses_perm_1", "SessionBusy"), ("ses_perm_1", "PermissionRequest")],
            )
        finally:
            package._pending_calls.pop("ses_perm_1").cancel()
            package._permission_queues.pop("ses_perm_1", None)
            package._session_state.pop("ses_perm_1", None)


class EvictAndKillSessionBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        package._session_state.clear()

    @patch("core.package._sbx_remove_sandbox", new_callable=AsyncMock)
    async def test_cancels_pending_call_and_clears_permission_queue(self, mock_remove_sandbox):
        package._session_state["ses_evict"] = ("orca-opencode-abc", "http://x", None)

        async def hang_forever():
            await asyncio.sleep(3600)

        task = asyncio.ensure_future(hang_forever())
        package._pending_calls["ses_evict"] = task
        package._get_permission_queue("ses_evict")

        await package._evict_and_kill_session_backend("ses_evict")

        self.assertNotIn("ses_evict", package._pending_calls)
        self.assertNotIn("ses_evict", package._permission_queues)
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-abc")
        with self.assertRaises(asyncio.CancelledError):
            await task


class AnswerPermissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        package._session_state.clear()

    def tearDown(self):
        package._session_state.pop("ses_ans", None)
        package._pending_calls.pop("ses_ans", None)
        package._permission_queues.pop("ses_ans", None)

    async def test_unknown_session_raises(self):
        with self.assertRaises(package.ToolError):
            await package._answer_permission("nope", "r1", "once", None, package.StatusHook())

    async def test_invalid_reply_raises(self):
        package._session_state["ses_ans"] = (None, "http://x", None)

        with self.assertRaises(package.ToolError):
            await package._answer_permission("ses_ans", "r1", "bogus", None, package.StatusHook())

    async def test_posts_reply_and_returns_ok_when_nothing_pending(self):
        package._session_state["ses_ans"] = (None, "http://x", None)
        mock_client = make_mock_client(FakeResponse(json_data={}))

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            result = await package._answer_permission(
                "ses_ans", "perm_1", "once", None, package.StatusHook()
            )

        self.assertEqual(result, {"session_id": "ses_ans", "status": "ok"})
        mock_client.post.assert_awaited_once_with(
            "http://x/session/ses_ans/permissions/perm_1",
            json={"response": "once"},
        )

    async def test_reply_error_status_raises(self):
        package._session_state["ses_ans"] = (None, "http://x", None)
        mock_client = make_mock_client(FakeResponse(status_code=500, text="boom"))

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                await package._answer_permission(
                    "ses_ans", "perm_1", "once", None, package.StatusHook()
                )

        self.assertIn("boom", str(ctx.exception))

    @patch("core.package._sbx_remove_sandbox", new_callable=AsyncMock)
    async def test_resumes_pending_call_and_returns_final_result(self, mock_remove_sandbox):
        package._session_state["ses_ans"] = ("orca-opencode-abc", "http://x", None)
        mock_client = make_mock_client(FakeResponse(json_data={}))
        status_hook = RecordingStatusHook()

        async def eventually_done():
            return {"info": {"finish": "stop"}, "parts": [{"type": "text", "text": "ok"}]}

        package._pending_calls["ses_ans"] = asyncio.ensure_future(eventually_done())

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            result = await package._answer_permission("ses_ans", "perm_1", "once", None, status_hook)

        self.assertEqual(
            result, {"session_id": "ses_ans", "response": "ok", "finish_reason": "stop"}
        )
        self.assertNotIn("ses_ans", package._pending_calls)
        # A finished run must tear its sandbox down instead of leaving it
        # running until the whole MCP process exits.
        self.assertNotIn("ses_ans", package._session_state)
        mock_remove_sandbox.assert_awaited_once_with("orca-opencode-abc")
        self.assertEqual(
            [event[:2] for event in status_hook.events],
            [("ses_ans", "SessionBusy"), ("ses_ans", "SessionIdle")],
        )

    async def test_resumed_call_hits_another_permission_request(self):
        package._session_state["ses_ans"] = (None, "http://x", None)
        mock_client = make_mock_client(FakeResponse(json_data={}))
        status_hook = RecordingStatusHook()

        async def hang_forever():
            await asyncio.sleep(3600)

        package._pending_calls["ses_ans"] = asyncio.ensure_future(hang_forever())
        package._get_permission_queue("ses_ans").put_nowait(
            {"id": "perm_2", "permission": "edit", "patterns": ["*.env"]}
        )

        with patch("core.package.httpx2.AsyncClient", return_value=mock_client):
            result = await package._answer_permission("ses_ans", "perm_1", "once", None, status_hook)

        self.assertEqual(result["status"], "permission_required")
        self.assertEqual(result["request_id"], "perm_2")
        # A chained permission request must not tear the sandbox down --
        # the session is still needed for the next answer_permission call.
        self.assertIn("ses_ans", package._session_state)
        self.assertEqual(
            [event[:2] for event in status_hook.events],
            [("ses_ans", "SessionBusy"), ("ses_ans", "PermissionRequest")],
        )
        package._pending_calls.pop("ses_ans").cancel()


class FakeMcp:
    def __init__(self):
        self.tools: dict[str, tuple] = {}

    def add_tool(self, fn, name=None, description=None):
        self.tools[name] = (fn, description)


class RegisterAgentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_calls_run_opencode_with_hooks(self):
        mcp = FakeMcp()
        visibility_hook = package.VisibilityHook()
        status_hook = package.StatusHook()
        package._register_agent_tool(mcp, "implementer", "Implements stuff", visibility_hook, status_hook)

        fn, description = mcp.tools["implementer"]
        self.assertEqual(description, "Implements stuff")

        with patch("core.package.run_opencode", new_callable=AsyncMock) as mock_run_opencode:
            mock_run_opencode.return_value = {
                "session_id": "ses_1",
                "response": "done",
                "finish_reason": "stop",
            }
            result = await fn(plan="do this", session_id="ses_1")

        self.assertEqual(result["response"], "done")
        call_kwargs = mock_run_opencode.await_args.kwargs
        self.assertEqual(call_kwargs["agent"], "implementer")
        self.assertEqual(call_kwargs["session_id"], "ses_1")
        self.assertEqual(call_kwargs["prompt"], "do this")
        self.assertIs(call_kwargs["visibility_hook"], visibility_hook)
        self.assertIs(call_kwargs["status_hook"], status_hook)

    async def test_failure_converts_to_tool_error(self):
        mcp = FakeMcp()
        package._register_agent_tool(
            mcp, "reviewer", "Reviews stuff", package.VisibilityHook(), package.StatusHook()
        )
        fn, _description = mcp.tools["reviewer"]

        with patch("core.package.run_opencode", new_callable=AsyncMock) as mock_run_opencode:
            mock_run_opencode.side_effect = RuntimeError("underlying failure")
            with self.assertRaises(package.ToolError) as ctx:
                await fn(plan="check this")

        self.assertIn("underlying failure", str(ctx.exception))


class RegisterAgentToolsTests(unittest.TestCase):
    def test_registers_answer_permission_alongside_discovered_agents(self):
        mcp = FakeMcp()
        with patch("core.package.discover_agents", return_value={"reviewer": "Reviews stuff"}):
            package.register_agent_tools(mcp)

        self.assertIn("reviewer", mcp.tools)
        self.assertIn("answer_permission", mcp.tools)

    def test_defaults_to_noop_hooks(self):
        mcp = FakeMcp()
        with patch("core.package.discover_agents", return_value={}):
            package.register_agent_tools(mcp)

        _fn, description = mcp.tools["answer_permission"]
        self.assertIn("permission_required", description)


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
            return package.discover_agents(project_dir)

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
