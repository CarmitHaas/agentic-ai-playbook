# MCP — Model Context Protocol

---

## Expose tools as thin adapters over the shared functions

**Problem.** An MCP server is another surface that needs your tools. Re-implementing the logic
there means two code paths that drift.

**Technique.** The FastMCP server imports the same pure functions the agent uses and wraps
each in a one-line `@mcp.tool`. The server has no business logic of its own — it's a transport
over the shared layer. (See *tools/ → one pure-function layer.*)

**When to use.** Whenever you want your agent's capabilities reachable by any MCP client
(Claude Desktop, another agent, a script).

**Code sketch.**
```python
mcp = FastMCP("bitext-customer-service-analyst")

@mcp.tool(description=COUNT_RECORDS_DESC)
def count_records(category=None, intent=None, keyword=None) -> dict:
    return analytics.count_records(category, intent, keyword).model_dump()

if __name__ == "__main__":
    mcp.run()        # stdio transport
```

**Verify it with a client.**
```python
from fastmcp import Client
async with Client("mcp_server/server.py") as c:           # launches server over stdio
    print(await c.call_tool("count_records", {"intent": "get_refund"}))  # -> 997
```

**Pitfall.** Reuse the same description constants the agent uses, so an MCP client and the
agent see identical tools. Confirm the MCP result matches the in-agent result — they should be
byte-identical because it's the same function.

**Source.** CS Data Analyst Agent — `mcp_server/server.py`, `mcp_server/client_demo.py`.
