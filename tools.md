# Tools — design

---

## One pure-function layer, many adapters

**Problem.** When the same capability is exposed to a LangChain agent and to an MCP server,
it's tempting to write the logic twice. The two copies drift, and neither is testable without
spinning up an LLM.

**Technique.** Put all logic in **framework-agnostic pure functions** (plain typed args in,
typed result out, no LangChain, no MCP). The LangChain `@tool` and the FastMCP `@mcp.tool` are
each a ~3-line adapter that calls the same function. One source of truth.

**When to use.** Any time a tool is reachable from more than one surface — or any time you
want to unit-test tools without an API key.

**Code sketch.**
```python
# analytics.py  (pure, testable, no LLM)
def count_records(category=None, intent=None) -> CountResult: ...

# tool_bindings.py            |   # mcp_server/server.py
@tool(args_schema=CountInput) |   @mcp.tool(description=COUNT_DESC)
def count_records(**kw):      |   def count_records(**kw):
    return analytics.count_records(**kw).model_dump()
```

**Pitfall.** Keep the *descriptions* as shared constants too, not just the logic — otherwise
the agent and the MCP server slowly describe the same tool differently.

**Source.** CS Data Analyst Agent — `tools/analytics.py` shared by `agent/` and `mcp_server/`.
11 tool tests run in 0.4s with no API key because of this split.

---

## Return counts and small samples, never whole tables

**Problem.** A tool that returns a filtered dataframe can dump thousands of rows into the
model's context. Slow, expensive, and it crowds out reasoning.

**Technique.** Design the return types to be small *by construction*. A `count` tool returns a
number; a `filter` tool returns the match count plus a handful of truncated examples — never
the full slice.

**When to use.** Any tool over a dataset or large collection.

**Code sketch.**
```python
class FilterResult(BaseModel):
    match_count: int                 # the number, always
    sample: list[RecordPreview]      # a few rows, fields truncated to ~240 chars
# count lives in its own tool so "how many" never pulls rows at all
```

**Pitfall.** Splitting "filter" and "count" into separate tools also gives you cleaner
multi-step chains (`filter -> count`) that the model composes naturally.

**Source.** CS Data Analyst Agent — `tools/schemas.py`, `tools/analytics.py`.

---

## Tool descriptions are part of the logic

**Problem.** A correct tool with a vague description never gets called at the right time. The
model only has the name, the description, and the schema to go on.

**Technique.** Write action-oriented descriptions that say **when** to use the tool and how it
chains with others, with concrete examples. Treat the description with the same care as the
code.

**When to use.** Every tool.

**Code sketch.**
```python
COUNT_RECORDS_DESC = (
  "Count how many records match a category/intent/keyword, as a number and a percentage. "
  "This is the counting half of a chain: to answer 'how many refund requests?', pass "
  "intent='get_refund'. Returns no rows, so it is cheap on large matches."
)
```

**Pitfall.** "If a human can't tell when to use a tool from its description alone, neither can
the model." Test your descriptions by reading them cold.

**Source.** CS Data Analyst Agent — `tools/schemas.py`.
