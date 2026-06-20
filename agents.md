# Agents — graph design & control flow

---

## Route before you act

**Problem.** If the agent decides which tool to call *and* whether a question is even in
scope in the same step, it leaks: off-topic questions get answered from the model's general
knowledge, and the system prompt fills with conditional instructions.

**Technique.** A dedicated **router node** runs first and classifies the request
(`structured` / `unstructured` / `out_of_scope` / `recommend`) before any tool is chosen.
Downstream nodes are simple because the decision is already made.

**When to use.** Any agent that has to handle more than one *kind* of request, or that must
refuse some requests.

**Code sketch.**
```python
builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_from_router,
    {"decline": "decline", "recommend": "recommend", "agent": "agent"})
```

**Pitfall.** Give the router conversation context, not just the last message, or follow-ups
like "what about refunds?" get misrouted. A small fast model is enough for this.

**Source.** CS Data Analyst Agent — `agent/router.py`.

---

## Refuse out-of-scope *structurally*, not with a prompt

**Problem.** "Only answer questions about X" in the system prompt is not a guardrail. The
model can be talked past it, and it still *sees* the off-topic question as answerable.

**Technique.** Route out-of-scope questions to a dedicated `decline` node that returns a
fixed refusal and goes straight to `END`. The generation model is never asked the question,
so there is nothing to jailbreak.

**When to use.** Any agent with a scope boundary (a data source, a domain, a policy).

**Code sketch.**
```python
def decline_node(state):                      # no LLM call at all
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}
# router: out_of_scope -> decline -> END
```

**Pitfall.** Don't also let the router be too eager — default to *in-scope* when unsure, or
you'll decline valid questions. Refuse only when clearly unrelated.

**Source.** CS Data Analyst Agent — `agent/graph.py` (`decline` node).

---

## Bound the loop: a state counter *and* a recursion-limit backstop

**Problem.** A ReAct loop can spin forever (re-calling tools, never finalizing). One guard is
not enough: LangGraph's own `recursion_limit` (default 25) can fire *before* your graceful
fallback and throw an ugly exception.

**Technique.** Track an `iterations` counter in state; at the cap, route to a `fallback` node
that returns a graceful message. Separately set the graph `recursion_limit` high enough that
your fallback always fires first.

**When to use.** Every tool-calling loop.

**Code sketch.**
```python
def route_after_agent(state):
    if not last_has_tool_calls(state): return "profile_update"
    if state["iterations"] >= MAX_ITERATIONS: return "fallback"   # business-level
    return "tools"
# at invoke time:
config = {"recursion_limit": 2 * MAX_ITERATIONS + 12}             # framework backstop
```

**Pitfall.** The two limits are different things. With `MAX_ITERATIONS=12` the default
recursion_limit of 25 trips first — you never see your nice fallback. Raise it.

**Source.** CS Data Analyst Agent — `agent/graph.py`, `main.py`. Hit this in testing.

---

## "Suggest, don't run": confirm-to-execute with a no-tools model

**Problem.** A feature like "what should I query next?" must *propose* a query, let the user
refine it, and only run it on confirmation. If you leave that decision to the router each
turn, it's unreliable — it executes the refinement instead of re-proposing.

**Technique.** Two things. (1) The recommender node uses the model **with no tools bound**, so
it physically cannot execute — only suggest. (2) A `pending_suggestion` flag in state makes
the next turn deterministic: a confirmation runs the suggestion; anything else refines it.

**When to use.** Any "human approves before the agent acts" flow.

**Code sketch.**
```python
llm_plain = get_generation_llm()              # NO .bind_tools  -> can't execute
# router, while a suggestion is pending:
if state.get("pending_suggestion"):
    return {"route": "structured" if is_confirmation(msg) else "recommend",
            "pending_suggestion": not is_confirmation(msg)}
```

**Pitfall.** Don't rely on the LLM router to tell "refine" from "confirm" every turn; a tiny
state flag + a keyword check is both cheaper and more predictable.

**Source.** CS Data Analyst Agent — Bonus B, `agent/graph.py` + `router.py`.

---

## Stream the reasoning, not just the answer

**Problem.** A black-box "here's the answer" agent is impossible to trust or debug.

**Technique.** Stream the graph with `stream_mode="updates"` and render each node's
contribution: the route, every tool call, every observation, then the final answer.

**When to use.** Always during development; in the CLI/UI it doubles as the "show your work"
the grader (and users) want.

**Code sketch.**
```python
for chunk in graph.stream(state, config, stream_mode="updates"):
    for node, update in chunk.items():
        if not update:               # nodes that change no state stream None
            continue
        render(node, update)
```

**Pitfall.** Nodes that return `{}` (e.g. a profile-update side effect) stream a `None`
update — guard for it or you crash mid-turn.

**Source.** CS Data Analyst Agent — `main.py`, `ui/streamlit_app.py`. Crashed on the `None`
update first time through.
