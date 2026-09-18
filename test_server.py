import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import server
from server import (
    extract_orca_terminal_handle,
    parse_json_output,
)


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


class ExtractOrcaTerminalHandleTests(unittest.TestCase):
    def test_valid_nested_handle(self):
        result = {"result": {"terminal": {"handle": "term_1"}}}
        self.assertEqual(extract_orca_terminal_handle(result), "term_1")

    def test_missing_handle_raises(self):
        with self.assertRaises(RuntimeError):
            extract_orca_terminal_handle({"result": {"terminal": {}}})

    def test_missing_terminal_raises(self):
        with self.assertRaises(RuntimeError):
            extract_orca_terminal_handle({"result": {}})

    def test_missing_result_raises(self):
        with self.assertRaises(RuntimeError):
            extract_orca_terminal_handle({})

    def test_non_dict_result_does_not_crash(self):
        with self.assertRaises(RuntimeError):
            extract_orca_terminal_handle({"result": "not-a-dict"})


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


# ---------------------------------------------------------------------------
# opencode serve process management.
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


class EnsureOpencodeServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._opencode_servers.clear()

    async def test_starts_and_caches_by_project_dir(self):
        process = FakeProcess([b"opencode server listening on http://127.0.0.1:4096\n"])

        with patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            base_url = await server.ensure_opencode_server(Path("/tmp/orca-bridge-proj-1"))

        self.assertEqual(base_url, "http://127.0.0.1:4096")
        self.assertIn(Path("/tmp/orca-bridge-proj-1"), server._opencode_servers)

    async def test_reuses_cached_server_when_still_alive(self):
        process = FakeProcess([b"opencode server listening on http://127.0.0.1:4096\n"])
        mock_exec = AsyncMock(return_value=process)

        with patch("server.asyncio.create_subprocess_exec", new=mock_exec):
            await server.ensure_opencode_server(Path("/tmp/orca-bridge-proj-2"))
            await server.ensure_opencode_server(Path("/tmp/orca-bridge-proj-2"))

        mock_exec.assert_awaited_once()

    async def test_respawns_when_cached_process_died(self):
        dead_process = FakeProcess([], returncode=1)
        alive_process = FakeProcess([b"opencode server listening on http://127.0.0.1:4097\n"])
        server._opencode_servers[Path("/tmp/orca-bridge-proj-3")] = (
            dead_process,
            "http://127.0.0.1:4096",
        )

        with patch(
            "server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=alive_process)
        ):
            base_url = await server.ensure_opencode_server(Path("/tmp/orca-bridge-proj-3"))

        self.assertEqual(base_url, "http://127.0.0.1:4097")

    @patch("server.os.killpg")
    @patch("server.os.getpgid", return_value=1)
    async def test_process_exits_without_listening_line_raises(self, mock_getpgid, mock_killpg):
        process = FakeProcess([b"some startup noise\n"], returncode=1)

        with patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            with self.assertRaises(RuntimeError):
                await server.ensure_opencode_server(Path("/tmp/orca-bridge-proj-4"))

    @patch("server.os.killpg")
    @patch("server.os.getpgid", return_value=1)
    @patch("server.OPENCODE_SERVE_STARTUP_TIMEOUT_SECONDS", 0.05)
    async def test_timeout_raises_and_kills_process(self, mock_getpgid, mock_killpg):
        process = FakeProcess(hang=True)

        with patch("server.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)):
            with self.assertRaises(asyncio.TimeoutError):
                await server.ensure_opencode_server(Path("/tmp/orca-bridge-proj-5"))

        mock_killpg.assert_called_once()


# ---------------------------------------------------------------------------
# Visibility terminal (best-effort, never allowed to break the data flow).
# ---------------------------------------------------------------------------


class EnsureVisibleTerminalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server._session_terminals.clear()

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_does_not_recreate_when_terminal_alive(self, mock_run_command):
        server._session_terminals["ses_1"] = "term_existing"
        mock_run_command.return_value = (0, "{}", "")

        await server.ensure_visible_terminal("http://x", "ses_1", "reviewer", Path("/tmp/p"))

        mock_run_command.assert_awaited_once()
        args = mock_run_command.await_args.args[0]
        self.assertIn("show", args)
        self.assertEqual(server._session_terminals["ses_1"], "term_existing")

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_creates_terminal_when_none_tracked(self, mock_run_command):
        create_output = json.dumps({"result": {"terminal": {"handle": "term_new"}}})
        mock_run_command.return_value = (0, create_output, "")

        await server.ensure_visible_terminal("http://x:1", "ses_2", "implementer", Path("/tmp/p"))

        self.assertEqual(server._session_terminals["ses_2"], "term_new")
        args = mock_run_command.await_args.args[0]
        self.assertIn("create", args)
        self.assertIn("opencode attach http://x:1 --session ses_2", " ".join(args))

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_recreates_when_tracked_terminal_is_gone(self, mock_run_command):
        server._session_terminals["ses_3"] = "term_stale"
        create_output = json.dumps({"result": {"terminal": {"handle": "term_fresh"}}})
        mock_run_command.side_effect = [
            (1, "", "not found"),
            (0, create_output, ""),
        ]

        await server.ensure_visible_terminal("http://x", "ses_3", "reviewer", Path("/tmp/p"))

        self.assertEqual(server._session_terminals["ses_3"], "term_fresh")

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_failure_is_swallowed_never_raises(self, mock_run_command):
        mock_run_command.side_effect = RuntimeError("orca is not running")

        await server.ensure_visible_terminal("http://x", "ses_4", "reviewer", Path("/tmp/p"))

        self.assertNotIn("ses_4", server._session_terminals)


# ---------------------------------------------------------------------------
# run_opencode orchestration.
# ---------------------------------------------------------------------------


class RunOpencodeTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_visible_terminal", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server.ensure_opencode_server", new_callable=AsyncMock)
    async def test_creates_new_session_when_none_given(
        self, mock_ensure_server, mock_create_session, mock_ensure_terminal, mock_send_prompt
    ):
        mock_ensure_server.return_value = "http://127.0.0.1:4096"
        mock_create_session.return_value = "ses_new"
        mock_send_prompt.return_value = {
            "info": {"finish": "stop"},
            "parts": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}],
        }

        result = await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertEqual(
            result, {"session_id": "ses_new", "response": "hello world", "finish_reason": "stop"}
        )
        mock_create_session.assert_awaited_once()
        mock_send_prompt.assert_awaited_once_with(
            "http://127.0.0.1:4096", "ses_new", "reviewer", "do it",
            server.OPENCODE_PROMPT_TIMEOUT_SECONDS,
        )

    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_visible_terminal", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server.ensure_opencode_server", new_callable=AsyncMock)
    async def test_reuses_given_session_id(
        self, mock_ensure_server, mock_create_session, mock_ensure_terminal, mock_send_prompt
    ):
        mock_ensure_server.return_value = "http://x"
        mock_send_prompt.return_value = {"info": {"finish": "stop"}, "parts": []}

        result = await server.run_opencode(agent="reviewer", prompt="do it", session_id="ses_existing")

        self.assertEqual(result["session_id"], "ses_existing")
        mock_create_session.assert_not_awaited()

    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_visible_terminal", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server.ensure_opencode_server", new_callable=AsyncMock)
    async def test_error_in_info_raises(
        self, mock_ensure_server, mock_create_session, mock_ensure_terminal, mock_send_prompt
    ):
        mock_ensure_server.return_value = "http://x"
        mock_create_session.return_value = "ses_1"
        mock_send_prompt.return_value = {"info": {"error": {"message": "boom"}}, "parts": []}

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="do it")

        self.assertIn("boom", str(ctx.exception))

    @patch("server.send_opencode_prompt", new_callable=AsyncMock)
    @patch("server.ensure_visible_terminal", new_callable=AsyncMock)
    @patch("server.create_opencode_session", new_callable=AsyncMock)
    @patch("server.ensure_opencode_server", new_callable=AsyncMock)
    async def test_non_text_parts_are_ignored(
        self, mock_ensure_server, mock_create_session, mock_ensure_terminal, mock_send_prompt
    ):
        mock_ensure_server.return_value = "http://x"
        mock_create_session.return_value = "ses_1"
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


class ToolWrapperTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.run_opencode", new_callable=AsyncMock)
    async def test_review_converts_failure_to_tool_error(self, mock_run_opencode):
        mock_run_opencode.side_effect = RuntimeError("underlying failure")

        with self.assertRaises(server.ToolError) as ctx:
            await server.review(plan="check this")

        self.assertIn("underlying failure", str(ctx.exception))

    @patch("server.run_opencode", new_callable=AsyncMock)
    async def test_review_passes_through_success(self, mock_run_opencode):
        mock_run_opencode.return_value = {
            "session_id": "ses_1",
            "response": "looks good",
            "finish_reason": "stop",
        }

        result = await server.review(plan="check this", session_id="ses_1")

        self.assertEqual(result["response"], "looks good")
        call_kwargs = mock_run_opencode.await_args.kwargs
        self.assertEqual(call_kwargs["agent"], "reviewer")
        self.assertEqual(call_kwargs["session_id"], "ses_1")
        self.assertIn("check this", call_kwargs["prompt"])

    @patch("server.run_opencode", new_callable=AsyncMock)
    async def test_implement_converts_failure_to_tool_error(self, mock_run_opencode):
        mock_run_opencode.side_effect = RuntimeError("underlying failure")

        with self.assertRaises(server.ToolError):
            await server.implement(plan="do this")

    @patch("server.run_opencode", new_callable=AsyncMock)
    async def test_implement_passes_through_success(self, mock_run_opencode):
        mock_run_opencode.return_value = {"response": "done"}

        result = await server.implement(plan="do this")

        self.assertEqual(result["response"], "done")

    async def test_review_browser_raises_tool_error(self):
        with self.assertRaises(server.ToolError):
            await server.review_browser(test_plan="x", account="demo")


if __name__ == "__main__":
    unittest.main()
