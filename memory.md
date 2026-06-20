# Memory — what an agent remembers

---

## Two different memories: episodic vs semantic

**Problem.** "Add memory" is ambiguous and people build one thing when they need two.
Remembering *this conversation* and remembering *who the user is* are different mechanisms.

**Technique.** Keep them separate.
- **Episodic** = the conversation transcript, persisted by a checkpointer keyed by a session
  id. Restores a chat across restarts and resolves follow-ups.
- **Semantic** = a small, *distilled* profile of the user (name, interests, preferences),
  stored separately and injected into the prompt. Not a replay of messages — facts.

**When to use.** Episodic for any multi-turn agent; add semantic when answers should
personalize or when "what do you remember about me?" must work.

**Code sketch.**
```python
# episodic: handled by the checkpointer, keyed per session
config = {"configurable": {"thread_id": session_id}}
# semantic: a separate per-user file, distilled after each answered turn
profiles/<user>.md   ->  ## Name / ## Interests / ## Preferences
```

**Pitfall.** Store the profile *outside* the checkpoint (its own file/table), keyed by user
not session — one user has many sessions, and the profile should follow them across all of
them.

**Source.** CS Data Analyst Agent — `agent/persistence.py` (episodic), `agent/profile.py`
(semantic).

---

## Persist with SqliteSaver, not MemorySaver

**Problem.** `MemorySaver` is in-memory only. Everything works in the demo, then the
conversation vanishes on restart — exactly what "persistent memory" was supposed to prevent.

**Technique.** Use `SqliteSaver` (or Postgres) so checkpoints land on disk. Same session id
after a restart restores the full history.

**When to use.** Any agent that should survive a process restart, or run across CLI + UI.

**Code sketch.**
```python
conn = sqlite3.connect(db_path, check_same_thread=False)   # shared across threads
saver = SqliteSaver(conn); saver.setup()
graph = build_graph(checkpointer=saver)
```

**Pitfall.** `check_same_thread=False` matters — Streamlit reruns on other threads. And cache
the saver/graph with `@st.cache_resource` so you don't reopen the DB every rerun.

**Source.** CS Data Analyst Agent — `agent/persistence.py`, `ui/streamlit_app.py`.

---

## Models deny their own memory — frame injected context as fact

**Problem.** I injected the user's profile into the system prompt, then asked "what do you
remember about me?" — and Llama-3.3 answered *"I don't retain information from previous
conversations."* The data was right there in context; the model reflexively refused it.

**Technique.** Frame injected memory assertively. Label it as **known, true facts** and
explicitly forbid the "I have no memory" denial.

**When to use.** Whenever you inject retrieved state (profile, prior session, RAG context)
and need the model to *own* it rather than disclaim it.

**Code sketch.**
```text
--- MEMORY: known facts about the current user ---
These facts are true and are your stored memory of this user. When asked what you
remember, answer from them. Do NOT claim you have no memory.
## Name: Carmit  ## Interests: refunds, cancellations ...
```

**Pitfall.** A soft framing ("here is some context") isn't enough — the safety-trained
reflex wins. Be explicit, and also tell it *not to call a tool* for profile questions (mine
hallucinated a `get_memory()` tool before I added that).

**Source.** CS Data Analyst Agent — `agent/profile.py`. Pure debugging discovery.
