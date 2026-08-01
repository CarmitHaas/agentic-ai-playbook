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

## Let the route pick the prompt, not just the path

**Problem.** One system prompt that covers every kind of request grows conditional
instructions ("if the question is structured do X, if open-ended do Y...") until the model
starts mixing behaviors — quoting exact counts in summaries, or padding count answers with
invented color.

**Technique.** Reuse the router's classification a second time: keep a small shared base
prompt and append a short route-specific hint at the agent node. The structured route gets
"use count/list tools, give exact numbers"; the unstructured route gets "pull a grounded
sample, then summarize in your own words". Each request sees only the instructions for its
own kind.

**When to use.** Any agent behind a router whose branches want different answer styles or
tool habits.

**Code sketch.**
```python
def _system_text_for_route(route):
    text = BASE_SYSTEM
    if route == "structured":   text += STRUCTURED_HINT
    elif route == "unstructured": text += UNSTRUCTURED_HINT
    return text            # decline/recommend never reach this prompt at all
```

**Pitfall.** The hint belongs at the agent node, not in the router — the router only labels.
Keep hints to a sentence or two; if a hint grows paragraphs, that branch wants its own node.

**Source.** CS Data Analyst Agent — `agent/graph.py` (`_system_text_for_route`). The grader
called out the clean split explicitly.

---

## A fallback path drifts from its primary unless they share one source of truth

**Problem.** A robust node often has a primary path and a fallback (typed structured output,
falling back to plain-text parsing). The primary is driven by a schema; the fallback prompt
enumerates the options *by hand*. Add an option later and only the primary gets it — the
fallback silently degrades, and because the primary almost always works, nothing ever fails
loudly.

**Technique.** Generate every hand-written enumeration from the same typed definition the
primary uses: one `Literal`/enum, and both the structured-output schema and the fallback
prompt text derive from it.

**When to use.** Anywhere a prompt restates something a schema already defines — route names,
tool names, status values.

**Code sketch.**
```python
Route = Literal["structured", "unstructured", "out_of_scope", "recommend"]

PLAIN_SUFFIX = "Reply with exactly one word: " + ", ".join(get_args(Route)) + "."
# instead of a hand-typed "structured, unstructured, or out_of_scope"  <- missed the 4th
```

**Pitfall.** This class of bug survives grading and demos: my router's fallback prompt still
listed three routes after I added the fourth (`recommend`), and the repo took 120/120 with
the bug latent — the structured-output primary always succeeded, so the stale fallback never
ran. I only caught it re-reading the file in a post-grade QA pass.

**Source.** CS Data Analyst Agent — `agent/router.py` (`ROUTER_SYSTEM_PLAIN` vs `Route`).

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

---

## Template prompts with `.replace`, not `.format`, when you inject data

**Problem.** Building a prompt with `PROMPT.format(schema=..., result=...)` blows up the moment the
injected data contains a brace. Real DB rows and free-text cells contain `{` / `}`, and `str.format`
reads those as field references — `KeyError`/`ValueError` — which the API wrapper turns into a 500 on
*specific* questions only.

**Technique.** Substitute placeholders with chained `.replace("{schema}", schema)`. `.replace` treats
the value as a literal, so braces in the data are harmless. Keep `.format` only where every
interpolated value is trusted (no user/DB text).

**When to use.** Any prompt that interpolates retrieved rows, documents, tool outputs, or user text.

**Code sketch.**
```python
user = (VERIFY_USER
        .replace("{question}", state.question)
        .replace("{result}", execution.render()))   # rows may contain { } — .format would crash
```

**Pitfall.** It fails on *some* inputs, not all, so it sails through a 5-question smoke test and only
shows up on the unlucky question under load. I caught it firing 25 perf questions, not the first 5.

**Source.** Text-to-SQL vLLM SLO — `agent/graph.py` (verify/revise); the provided `schema.py` had the
same class of bug on a `None` foreign-key column.

---

## Structured verify verdict: parse defensively, default to the SAFE branch

**Problem.** A verify step that asks "is this answer ok?" gets back prose, fenced JSON, or half-JSON.
Parse it naively and a flaky reply silently routes the wrong way.

**Technique.** Ask for one-line `{"ok": bool, "issue": str}`. Parse defensively (strip a ```json fence,
regex the first `{...}`, `json.loads`, coerce types). On ANY parse failure default to *not ok*, so an
unparseable verdict triggers a revise (cheap, recoverable) instead of passing an unvalidated answer.

**When to use.** Any self-check / verify / guard node whose output drives control flow.

**Code sketch.**
```python
def _parse_verdict(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:    o = json.loads(m.group(0)); return bool(o.get("ok")), str(o.get("issue") or "")
    except Exception: return False, "unparseable verdict"   # safe default = revise
```

**Pitfall.** Choose the default deliberately — the cost of being wrong is asymmetric. Here "not ok"
costs one extra LLM call; "ok" would ship a wrong answer. Default toward the cheap mistake.

**Source.** Text-to-SQL vLLM SLO — `agent/graph.py` `verify_node` / `_parse_verdict`.
