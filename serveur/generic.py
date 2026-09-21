"""
Generic MCP server: the same OpenCode sandboxing trunk (core/package.py) as
serveur/orca.py, with no integration plugged in -- no visible pane, no
status reporting anywhere. Use this outside of Orca.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer

from core import package

mcp = MCPServer("opencode-bridge")

package.register_agent_tools(mcp)


if __name__ == "__main__":
    mcp.run()
