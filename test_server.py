import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import server
from server import (
    build_opencode_command,
    extract_event_error,
    extract_event_text,
    extract_exit_marker_code,
    extract_orca_terminal_handle,
    get_step_finish_reason,
    make_exit_marker_prefix,
    parse_json_output,
    parse_opencode_event,
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


class BuildOpencodeCommandTests(unittest.TestCase):
    def test_without_session_id(self):
        command = build_opencode_command(
            agent="reviewer", prompt="hello world", exit_marker_token="tok1"
        )
        self.assertTrue(
            command.startswith(
                "opencode run --agent reviewer --format json 'hello world'"
            )
        )

    def test_with_session_id(self):
        command = build_opencode_command(
            agent="implementer",
            prompt="do the thing",
            exit_marker_token="tok2",
            session_id="ses_123",
        )
        self.assertTrue(
            command.startswith(
                "opencode run --agent implementer --format json --session ses_123 'do the thing'"
            )
        )

    def test_prompt_is_shell_escaped(self):
        command = build_opencode_command(
            agent="reviewer", prompt="it's a test", exit_marker_token="tok3"
        )
        self.assertIn("'it'\"'\"'s a test'", command)

    def test_appends_exit_marker_suffix(self):
        command = build_opencode_command(
            agent="reviewer", prompt="hi", exit_marker_token="abc123"
        )
        self.assertIn("ORCA_BRIDGE_EXIT_abc123:", command)
        self.assertIn('"$?"', command)

    def test_session_id_with_shell_metacharacters_is_escaped(self):
        malicious = "ses_1; rm -rf /; $(whoami)"
        command = build_opencode_command(
            agent="reviewer",
            prompt="hi",
            exit_marker_token="tok",
            session_id=malicious,
        )
        # shlex.join must quote the whole token as one shell word.
        self.assertIn(shlex.quote(malicious), command)


class ExtractExitMarkerCodeTests(unittest.TestCase):
    def test_matches_exit_code(self):
        prefix = make_exit_marker_prefix("tok")
        self.assertEqual(extract_exit_marker_code(f"{prefix}0", prefix), "0")

    def test_matches_nonzero_exit_code_with_surrounding_text(self):
        prefix = make_exit_marker_prefix("tok")
        self.assertEqual(
            extract_exit_marker_code(f"noise {prefix}127 trailing", prefix), "127"
        )

    def test_no_match_returns_none(self):
        prefix = make_exit_marker_prefix("tok")
        self.assertIsNone(extract_exit_marker_code("nothing here", prefix))

    def test_different_tokens_do_not_cross_match(self):
        # A marker for a different run's token (or arbitrary file content
        # containing another run's marker) must not match this run's.
        other_run_prefix = make_exit_marker_prefix("other-token")
        this_run_prefix = make_exit_marker_prefix("this-token")
        self.assertIsNone(
            extract_exit_marker_code(f"{other_run_prefix}0", this_run_prefix)
        )


class ParseOpencodeEventTests(unittest.TestCase):
    def test_valid_event(self):
        self.assertEqual(parse_opencode_event('{"type": "text"}'), {"type": "text"})

    def test_invalid_json_returns_none(self):
        self.assertIsNone(parse_opencode_event("not json"))

    def test_non_dict_json_returns_none(self):
        self.assertIsNone(parse_opencode_event("[1, 2, 3]"))


class ExtractEventTextTests(unittest.TestCase):
    def test_extracts_text(self):
        event = {"type": "text", "part": {"text": "hello"}}
        self.assertEqual(extract_event_text(event), "hello")

    def test_wrong_type_returns_none(self):
        event = {"type": "step_finish", "part": {"text": "hello"}}
        self.assertIsNone(extract_event_text(event))

    def test_missing_part_returns_none(self):
        self.assertIsNone(extract_event_text({"type": "text"}))

    def test_empty_text_returns_none(self):
        event = {"type": "text", "part": {"text": ""}}
        self.assertIsNone(extract_event_text(event))


class GetStepFinishReasonTests(unittest.TestCase):
    def test_stop_reason(self):
        event = {"type": "step_finish", "part": {"reason": "stop"}}
        self.assertEqual(get_step_finish_reason(event), "stop")

    def test_length_reason_is_terminal(self):
        event = {"type": "step_finish", "part": {"reason": "length"}}
        self.assertEqual(get_step_finish_reason(event), "length")

    def test_tool_calls_reason_is_not_terminal(self):
        # A tool-calls step_finish means more steps follow (after the tool
        # runs); the loop must keep polling, not return early.
        event = {"type": "step_finish", "part": {"reason": "tool-calls"}}
        self.assertIsNone(get_step_finish_reason(event))

    def test_wrong_type_returns_none(self):
        event = {"type": "text", "part": {"reason": "stop"}}
        self.assertIsNone(get_step_finish_reason(event))

    def test_missing_part_returns_none(self):
        self.assertIsNone(get_step_finish_reason({"type": "step_finish"}))


class ExtractEventErrorTests(unittest.TestCase):
    def test_string_error(self):
        event = {"type": "error", "error": "boom"}
        self.assertEqual(extract_event_error(event), "boom")

    def test_dict_error(self):
        event = {"type": "error", "error": {"message": "boom"}}
        self.assertIn("boom", extract_event_error(event))

    def test_non_error_type_returns_none(self):
        event = {"type": "text", "error": "boom"}
        self.assertIsNone(extract_event_error(event))


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


class ReadOrcaTerminalTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_nonzero_returncode_raises(self, mock_run_command):
        mock_run_command.return_value = (1, "", "boom")

        with self.assertRaises(RuntimeError) as ctx:
            await server.read_orca_terminal("term_x")

        self.assertIn("boom", str(ctx.exception))

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_success_returns_parsed_json(self, mock_run_command):
        mock_run_command.return_value = (0, '{"ok": true}', "")

        result = await server.read_orca_terminal("term_x")

        self.assertEqual(result, {"ok": True})


def make_terminal_read_output(tail: list[str]) -> str:
    return json.dumps({"result": {"terminal": {"tail": tail}}})


CREATE_TERMINAL_OUTPUT = json.dumps(
    {"result": {"terminal": {"handle": "term_test"}}}
)


class RunOpencodeTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_mid_run_read_failure_closes_terminal(self, mock_run_command):
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (1, "", "orca read boom"),
            (0, "{}", ""),  # the best-effort close call
        ]

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertIn("orca read boom", str(ctx.exception))
        self.assertEqual(mock_run_command.await_count, 3)
        close_call = mock_run_command.await_args_list[2]
        self.assertIn("close", close_call.args[0])

    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_dedups_repeated_lines_and_stops_on_finish(
        self, mock_run_command, mock_sleep
    ):
        text_event = json.dumps({"type": "text", "part": {"text": "hello "}})
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (0, make_terminal_read_output([text_event]), ""),
            (
                0,
                make_terminal_read_output(
                    [
                        text_event,  # already seen: must not be appended again
                        json.dumps({"type": "text", "part": {"text": "world"}}),
                        json.dumps(
                            {"type": "step_finish", "part": {"reason": "stop"}}
                        ),
                    ]
                ),
                "",
            ),
        ]

        result = await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertEqual(result["response"], "hello world")
        self.assertEqual(result["terminal_handle"], "term_test")
        self.assertEqual(result["finish_reason"], "stop")
        mock_sleep.assert_awaited_once_with(1.0)

    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_tool_calls_finish_reason_does_not_end_run(
        self, mock_run_command, mock_sleep
    ):
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (
                0,
                make_terminal_read_output(
                    [json.dumps({"type": "step_finish", "part": {"reason": "tool-calls"}})]
                ),
                "",
            ),
            (
                0,
                make_terminal_read_output(
                    [
                        json.dumps({"type": "text", "part": {"text": "final answer"}}),
                        json.dumps({"type": "step_finish", "part": {"reason": "stop"}}),
                    ]
                ),
                "",
            ),
        ]

        result = await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertEqual(result["response"], "final answer")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(mock_run_command.await_count, 3)

    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_non_stop_finish_reason_still_ends_run(
        self, mock_run_command, mock_sleep
    ):
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (
                0,
                make_terminal_read_output(
                    [json.dumps({"type": "step_finish", "part": {"reason": "length"}})]
                ),
                "",
            ),
        ]

        result = await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertEqual(result["finish_reason"], "length")

    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_error_event_raises(self, mock_run_command, mock_sleep):
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (
                0,
                make_terminal_read_output(
                    [json.dumps({"type": "error", "error": "boom"})]
                ),
                "",
            ),
            (0, "{}", ""),  # the best-effort close call
        ]

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertIn("boom", str(ctx.exception))
        self.assertEqual(mock_run_command.await_count, 3)
        close_call = mock_run_command.await_args_list[2]
        self.assertIn("close", close_call.args[0])

    @patch("server.uuid.uuid4")
    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_nonzero_exit_without_finish_event_raises(
        self, mock_run_command, mock_sleep, mock_uuid4
    ):
        mock_uuid4.return_value.hex = "faketoken"
        marker = make_exit_marker_prefix("faketoken")
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (
                0,
                make_terminal_read_output(
                    [
                        json.dumps({"type": "text", "part": {"text": "partial"}}),
                        "! permission requested: external_directory; auto-rejecting",
                        f"{marker}1",
                    ]
                ),
                "",
            ),
            (0, "{}", ""),  # the best-effort close call
        ]

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertIn("exit_code=1", str(ctx.exception))
        self.assertIn("partial", str(ctx.exception))
        self.assertEqual(mock_run_command.await_count, 3)

    @patch("server.uuid.uuid4")
    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_zero_exit_without_finish_event_returns_partial_success(
        self, mock_run_command, mock_sleep, mock_uuid4
    ):
        # The JSONL stream can legitimately miss the final "stop" event
        # (bounded tail window scrolling past it), but a zero exit code
        # from the shell itself is still a reliable success signal.
        mock_uuid4.return_value.hex = "faketoken"
        marker = make_exit_marker_prefix("faketoken")
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (
                0,
                make_terminal_read_output(
                    [
                        json.dumps({"type": "text", "part": {"text": "done"}}),
                        f"{marker}0",
                    ]
                ),
                "",
            ),
        ]

        result = await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertEqual(result["response"], "done")
        self.assertEqual(result["finish_reason"], "process_exit")

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_timeout_raises_with_partial_response(self, mock_run_command):
        # Second call is the best-effort terminal close triggered by the timeout.
        mock_run_command.side_effect = [(0, CREATE_TERMINAL_OUTPUT, ""), (0, "{}", "")]

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="hi", timeout_seconds=0)

        self.assertIn("Timed out", str(ctx.exception))
        self.assertEqual(mock_run_command.await_count, 2)

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_timeout_closes_terminal(self, mock_run_command):
        mock_run_command.side_effect = [(0, CREATE_TERMINAL_OUTPUT, ""), (0, "{}", "")]

        with self.assertRaises(RuntimeError):
            await server.run_opencode(agent="reviewer", prompt="hi", timeout_seconds=0)

        close_call = mock_run_command.await_args_list[1]
        self.assertIn("close", close_call.args[0])
        self.assertIn("term_test", close_call.args[0])

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_timeout_close_failure_does_not_mask_original_error(
        self, mock_run_command
    ):
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            RuntimeError("close also failed"),
        ]

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="hi", timeout_seconds=0)

        self.assertIn("Timed out", str(ctx.exception))


class RunOpencodeAdditionalTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_malformed_read_payload_is_tolerated(
        self, mock_run_command, mock_sleep
    ):
        # A poll that doesn't have the expected result.terminal.tail shape
        # (e.g. "terminal": null) must not crash the loop -- it's skipped
        # as if there were no new lines, and polling continues.
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (0, json.dumps({"result": {"terminal": None}}), ""),
            (
                0,
                make_terminal_read_output(
                    [json.dumps({"type": "step_finish", "part": {"reason": "stop"}})]
                ),
                "",
            ),
        ]

        result = await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertEqual(result["finish_reason"], "stop")

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_terminal_create_failure_raises(self, mock_run_command):
        mock_run_command.side_effect = [(1, "", "boom")]

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertIn("boom", str(ctx.exception))

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_terminal_create_missing_handle_raises(self, mock_run_command):
        mock_run_command.side_effect = [
            (0, json.dumps({"result": {"terminal": {}}}), "")
        ]

        with self.assertRaises(RuntimeError):
            await server.run_opencode(agent="reviewer", prompt="hi")

    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_non_list_tail_is_ignored(self, mock_run_command, mock_sleep):
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (0, json.dumps({"result": {"terminal": {"tail": "not-a-list"}}}), ""),
            (
                0,
                make_terminal_read_output(
                    [json.dumps({"type": "step_finish", "part": {"reason": "stop"}})]
                ),
                "",
            ),
        ]

        result = await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertEqual(result["finish_reason"], "stop")

    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_session_id_detected_from_event(self, mock_run_command, mock_sleep):
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (
                0,
                make_terminal_read_output(
                    [
                        json.dumps(
                            {
                                "type": "text",
                                "part": {"text": "hi"},
                                "sessionID": "ses_abc",
                            }
                        ),
                        json.dumps({"type": "step_finish", "part": {"reason": "stop"}}),
                    ]
                ),
                "",
            ),
        ]

        result = await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertEqual(result["session_id"], "ses_abc")

    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_content_filter_reason_is_terminal(self, mock_run_command, mock_sleep):
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (
                0,
                make_terminal_read_output(
                    [
                        json.dumps(
                            {"type": "step_finish", "part": {"reason": "content_filter"}}
                        )
                    ]
                ),
                "",
            ),
        ]

        result = await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertEqual(result["finish_reason"], "content_filter")

    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_dict_error_event_raises_with_json_body(
        self, mock_run_command, mock_sleep
    ):
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (
                0,
                make_terminal_read_output(
                    [
                        json.dumps(
                            {"type": "error", "error": {"message": "boom", "code": 42}}
                        )
                    ]
                ),
                "",
            ),
            (0, "{}", ""),  # the best-effort close call
        ]

        with self.assertRaises(RuntimeError) as ctx:
            await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertIn("boom", str(ctx.exception))
        self.assertEqual(mock_run_command.await_count, 3)

    @patch("server.uuid.uuid4")
    @patch("server.asyncio.sleep", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_other_runs_marker_does_not_false_positive(
        self, mock_run_command, mock_sleep, mock_uuid4
    ):
        # A marker matching a *different* run's token -- e.g. echoed back
        # via a "read" tool call on a file that happens to contain it --
        # must not be mistaken for this run's own completion signal.
        mock_uuid4.return_value.hex = "thisrun"
        other_marker = make_exit_marker_prefix("otherrun")
        mock_run_command.side_effect = [
            (0, CREATE_TERMINAL_OUTPUT, ""),
            (
                0,
                make_terminal_read_output(
                    [
                        f"{other_marker}0",
                        json.dumps({"type": "text", "part": {"text": "real answer"}}),
                        json.dumps({"type": "step_finish", "part": {"reason": "stop"}}),
                    ]
                ),
                "",
            ),
        ]

        result = await server.run_opencode(agent="reviewer", prompt="hi")

        self.assertEqual(result["response"], "real answer")
        self.assertEqual(result["finish_reason"], "stop")


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
            "stderr": "",
            "terminal_handle": "term_1",
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


class CloseOrcaTerminalTests(unittest.IsolatedAsyncioTestCase):
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_success_path_calls_orca_close(self, mock_run_command):
        mock_run_command.return_value = (0, "{}", "")

        await server.close_orca_terminal("term_x")

        mock_run_command.assert_awaited_once()
        args = mock_run_command.await_args.args[0]
        self.assertIn("close", args)
        self.assertIn("term_x", args)

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_failure_is_swallowed(self, mock_run_command):
        mock_run_command.side_effect = RuntimeError("orca close boom")

        await server.close_orca_terminal("term_x")  # must not raise


if __name__ == "__main__":
    unittest.main()
