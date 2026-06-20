# LLM-ops — models & dependencies

---

## Small model for routing, large for generation

**Problem.** Using one big model for everything is slow and expensive. Most of an agent's LLM
calls are cheap jobs (classify a question, merge a short profile) that don't need the big
model — but they run on every turn.

**Technique.** Split by role. A small, fast model handles routing and profile distillation; a
large model handles tool-calling, reasoning, and writing. Mixture-of-Experts models (e.g.
~3B *active* params) make great routers — much cheaper per token than a dense 70B.

**When to use.** Any agent with high-volume "easy" calls (classification, extraction,
short summary merges) alongside the "hard" reasoning.

**Code sketch.**
```python
GEN_MODEL    = "meta-llama/Llama-3.3-70B-Instruct"        # reasoning, tools, writing
ROUTER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"         # routing + profile distillation
```

**Pitfall.** Justify the split in your README — graders and reviewers reward the reasoning,
and it documents intent for the next person (you).

**Source.** CS Data Analyst Agent — `agent/llm.py`, `config.py`.

---

## Verify model IDs against the live catalog; pin a lockfile

**Problem.** Hardcoding a model ID that "everyone uses" breaks when the provider's catalog
changes. I planned to use an 8B router model — it had been removed from the Nebius catalog by
the time I built.

**Technique.** Hit the provider's `/v1/models` endpoint and confirm the exact IDs *before*
pinning them. Keep model IDs in one config module, overridable via `.env`. Commit a lockfile
so the dependency tree is reproducible.

**When to use.** Every project, at setup and whenever a model call 404s.

**Code sketch.**
```python
client = OpenAI(api_key=KEY, base_url=BASE_URL)
ids = sorted(m.id for m in client.models.list().data)   # confirm before pinning
# deps: commit uv.lock / requirements.txt with exact versions
```

**Pitfall.** Catalogs drift faster than tutorials. The model in the assignment's example
notebook didn't exist anymore; the live list did.

**Source.** CS Data Analyst Agent — verified the catalog, then set `config.py`.
