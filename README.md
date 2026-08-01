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
- Template prompts with `.replace`, not `.format`, when injecting data
- Structured verify verdict, parsed defensively, default to the safe branch

### [tools/](tools.md) — tool design
- One pure-function layer, many adapters (agent + MCP share it)
- Return counts and small samples, never whole tables
- Tool descriptions are part of the logic
- The user's screenshots folder as an evidence conveyor (auth-proof human-in-the-loop capture)

### [memory/](memory.md) — what an agent remembers
- Two different memories: episodic vs semantic
- Persist with SqliteSaver, not MemorySaver
- Models deny their own memory — frame injected context as fact

### [llm-ops/](llm-ops.md) — models & dependencies
- Small model for routing, large for generation
- Verify model IDs against the live catalog; pin a lockfile
- Turn tracing off when measuring latency / SLOs
- MoE serving: memory by total params, compute by active params
- Build on a cheap same-model backend before the scarce GPU
- Read the operation error body, not the console label; pre-flight every quota the template allocates
- Napkin-math the communication budget before the first GPU-hour; pre-register the deciding log line
- Probe for silent degradation before the metered run — CUDA visible, right device, compile really fires
- Drive a notebook from a persistent kernel; write outputs back by cell id, never by index
- Check the library API at the installed version, not from memory (transformers 5.x CLIP)
- When the framework pins torch, pick the image by the wheel's CUDA major (driver 570 vs cu13 wheels)
- A CUDA image is not a build box: JIT serving stacks need a toolchain and the venv on PATH
- Reverse-engineer a spec-based CLI from its installed schemas, then prove it on the tool's own mock server

### [mcp/](mcp.md) — Model Context Protocol
- Expose tools as thin adapters over the shared functions

### [rag/](rag.md) — retrieval-augmented generation
- Measure retrieval separately (page_hit@k); it's usually the bottleneck
- Raise k before reaching for a reranker
- A domain-mismatched reranker makes things worse
- Stricter prompts trade correctness for faithfulness
- Cache every LLM call for reproducible experiments
- A nearest-neighbor index has no reject option (open-set forces a wrong match)
- Match the embedder to the distinction you need (CLIP embeds appearance, not identity)

### [evaluation/](evaluation.md) — rubrics & LLM-as-judge
- Reasoning before verdict; judge what you must, measure what you can
- Hard go/no-go gates, separate from the cumulative score
- Judges have a leniency bias; agreement tracks objectivity
- Isolate one variable at a time when improving
- Garbage source data masquerades as hallucination
- Execution accuracy on canonicalized row-sets (text-to-SQL)
- Per-iteration pass rate: prove the agent loop earns its keep
- Proof-carrying deliverables: CI re-derives the report from committed logs
- Every number in the report must trace to an output cell (same measurement conditions)
- Near a decision threshold, search the lever combination, not just single levers
- Compare runs at the point where they are still comparable (iteration 0, not end of run)
- A metric whose definition contains the knob you are turning is not a measurement
- Find out what dominates a metric before you use it as a proxy
- A penalty coefficient is only meaningful as a share of the objective
- Commit the prediction and its falsifier in writing before the run
- Pick evaluation data that is allowed to fail (a 100% score means the test is too easy)
- Write the analysis from captured outputs, not the outputs you expect
- A locked harness with a wrong number: recover it from the raw artifact and disclose the source
- Assert every claim in the writeup against the artifacts with a script, not a read-through

## Sources

- **CS Data Analyst Agent** — LangGraph ReAct agent over the Bitext dataset
  ([repo](https://github.com/CarmitHaas/customer-service-agent-carmit-haas)). `agents`, `tools`,
  `memory`, `llm-ops`, `mcp`.
- **RAG** — a FinanceBench pipeline (retrieval/reranking/faithfulness experiments). `rag`.
- **Evals** — LLM-as-judge over product descriptions (rubric, judge calibration, improvement
  loops). `evaluation`.
- **Text-to-SQL vLLM SLO** — a LangGraph text-to-SQL agent on BIRD-bench served by Qwen3-30B-A3B
  (vLLM), with Prometheus/Grafana/Langfuse observability and an SLO load test
  ([repo](https://github.com/CarmitHaas/text-to-sql-vllm-slo-carmit-haas)). `agents`, `evaluation`,
  `llm-ops`.
- **DDP Scaling Anatomy** — GPT-2 Large on 1 vs 4 H100s over TCP, with a live cloud-quota
  incident debugged mid-session
  ([repo](https://github.com/CarmitHaas/ddp-scaling-anatomy)). `tools`, `llm-ops`, `evaluation`.
- **Glass-box PPO** - PPO on GPT-2 built from scratch (no `PPOTrainer`), swept over clip epsilon,
  GAE lambda, KL beta, PPO epochs and a no-critic ablation: 11 controlled runs on CPU.
  `evaluation`, `llm-ops`.
- **roofline_to_Cuda** — GPU performance homework (roofline model, decode-loop optimization, torch.compile & CUDA graphs), executed on a rented H100. `llm-ops`, `evaluation`.
- **Quantization & serving** - fake quantization from first principles, then the same 7B model served
  BF16 vs on-the-fly FP8 on an H100 and benchmarked with guidellm (memory 1.74x, decode 32% faster,
  prefill 17% slower at batch size 1). `llm-ops`, `evaluation`.
- **Multimodal (BLIP + CLIP)** - image captioning, VQA, and CLIP face recognition on LFW, run locally on CPU with a train/test split and an out-of-set stranger test. `evaluation`, `rag`, `llm-ops`.
- [Nir Diamant — Agent Memory Techniques](https://github.com/NirDiamant/Agent_Memory_Techniques)
  (reference for the memory work).

---
*Template for new entries: [TEMPLATE.md](TEMPLATE.md).*
