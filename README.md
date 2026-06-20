# Agentic AI Playbook

My working notebook of techniques and best practices for building LLM agents — the lessons
that actually changed my results, written down so I can reuse them on any project, from
Claude Code or the web.

Each entry is a real lesson from a real build, not generic advice. Every one names where it
came from and the mistake it fixes.

## How I use this

- **In Claude Code / Cursor:** clone it next to whatever I'm building; the agent can read it.
- **On the web / mobile:** browse it on GitHub (a Notion mirror is planned — these files import
  straight into Notion as pages).
- **Quick recall:** the highest-leverage entries are also mirrored into my Claude Code memory
  so they surface automatically.

## Index

### [agents/](agents.md) — graph design & control flow
- Route before you act (a dedicated router node)
- Refuse out-of-scope **structurally**, not with a prompt
- Bound the loop: state counter *and* recursion-limit backstop
- "Suggest, don't run": confirm-to-execute with a no-tools model
- Stream the reasoning, not just the answer

### [tools/](tools.md) — tool design
- One pure-function layer, many adapters (agent + MCP share it)
- Return counts and small samples, never whole tables
- Tool descriptions are part of the logic

### [memory/](memory.md) — what an agent remembers
- Two different memories: episodic vs semantic
- Persist with SqliteSaver, not MemorySaver
- Models deny their own memory — frame injected context as fact

### [llm-ops/](llm-ops.md) — models & dependencies
- Small model for routing, large for generation
- Verify model IDs against the live catalog; pin a lockfile

### [mcp/](mcp.md) — Model Context Protocol
- Expose tools as thin adapters over the shared functions

### [rag/](rag.md) — retrieval-augmented generation
- Measure retrieval separately (page_hit@k); it's usually the bottleneck
- Raise k before reaching for a reranker
- A domain-mismatched reranker makes things worse
- Stricter prompts trade correctness for faithfulness
- Cache every LLM call for reproducible experiments

### [evaluation/](evaluation.md) — rubrics & LLM-as-judge
- Reasoning before verdict; judge what you must, measure what you can
- Hard go/no-go gates, separate from the cumulative score
- Judges have a leniency bias; agreement tracks objectivity
- Isolate one variable at a time when improving
- Garbage source data masquerades as hallucination

## Sources

- **CS Data Analyst Agent** — LangGraph ReAct agent over the Bitext dataset
  ([repo](https://github.com/CarmitHaas/customer-service-agent-carmit-haas)). `agents`, `tools`,
  `memory`, `llm-ops`, `mcp`.
- **RAG** — a FinanceBench pipeline (retrieval/reranking/faithfulness experiments). `rag`.
- **Evals** — LLM-as-judge over product descriptions (rubric, judge calibration, improvement
  loops). `evaluation`.
- [Nir Diamant — Agent Memory Techniques](https://github.com/NirDiamant/Agent_Memory_Techniques)
  (reference for the memory work).

---
*Template for new entries: [TEMPLATE.md](TEMPLATE.md).*
