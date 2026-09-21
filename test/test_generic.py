import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import package
from serveur import generic


class DefaultHooksAreNoOpsTests(unittest.IsolatedAsyncioTestCase):
    """serveur/generic.py wires no integration in at all, relying entirely on
    the trunk's own default VisibilityHook/StatusHook behaving as working
    no-ops."""

    async def test_visibility_hook_before_spawn_returns_none(self):
        self.assertIsNone(await package.VisibilityHook().before_spawn("reviewer", Path("/tmp/p")))

    async def test_visibility_hook_after_spawn_does_nothing(self):
        await package.VisibilityHook().after_spawn("handle", "http://x", "ses_1")  # must not raise

    async def test_visibility_hook_is_alive_defaults_to_true(self):
        self.assertTrue(await package.VisibilityHook().is_alive("handle"))

    def test_visibility_hook_extra_env_is_empty(self):
        self.assertEqual(package.VisibilityHook().extra_env("handle"), {})

    async def test_status_hook_post_status_does_nothing(self):
        await package.StatusHook().post_status("ses_1", "SessionBusy", {})  # must not raise


class GenericServerWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_registers_answer_permission_tool(self):
        tools = await generic.mcp.list_tools()
        self.assertIn("answer_permission", {tool.name for tool in tools})

    def test_server_name_is_generic(self):
        self.assertEqual(generic.mcp.name, "opencode-bridge")

    def test_module_never_imports_orca_integration(self):
        self.assertNotIn("orca", dir(generic))


if __name__ == "__main__":
    unittest.main()
