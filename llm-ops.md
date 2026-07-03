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

---

## Turn tracing OFF when you measure latency / SLOs

**Problem.** Agent tracing (Langfuse, etc.) is great for debugging and terrible for a clean latency
number. Left on during a load test it taxes every request (span build + enqueue on the hot path), and
when the trace backend runs on the same box its ingest contends with the model server. Your reported
P95 then includes a tracing tax you'd misattribute to the model.

**Technique.** Capture a handful of traces for the screenshots, then run the *measured* load test with
tracing disabled. If the handler is gated on env vars, just relaunch without them.

**When to use.** Any time the number you report is latency / throughput / SLO, not correctness.

**Code sketch.**
```bash
# handler is gated on these at import; unset -> handler=None -> callbacks=[]
LANGFUSE_PUBLIC_KEY= LANGFUSE_SECRET_KEY= uv run uvicorn agent.server:app --port 8001
```

**Pitfall.** Easy to leave on because the keys live in `.env` and the server "just works." A QA pass
caught that my ~6000-request load test would have shipped tracing-inflated SLO numbers.

**Source.** Text-to-SQL vLLM SLO — `agent/server.py` (handler gating) + the slot runbook.

---

## MoE serving: memory is set by TOTAL params, compute by ACTIVE params

**Problem.** Sizing a Mixture-of-Experts model like a dense one gives the wrong call — you either think
it won't fit or you under-provision KV cache.

**Technique.** "Big to store, small to run." All experts must be resident, so **memory** tracks total
params (Qwen3-30B-A3B: ~30.5B → ~61GB BF16, no room for KV on an 80GB card → quantize to FP8 ~30GB).
Per-token **compute** tracks active params (~3.3B), so decode is cheap and a low-latency SLO at real
concurrency is feasible on one GPU. The levers cluster on memory (weight + KV quant, max-model-len,
gpu-mem-util), not parallelism (tensor / expert-parallel are inert on a single GPU).

**When to use.** Serving or capacity-planning any MoE on a fixed GPU.

**Pitfall.** `--max-model-len` is the sneaky one: leaving the model's native 262K context makes vLLM
reserve KV for sequences you never send (prompts were ~3K), starving concurrency. Set it to your real
P99 prompt + output.

**Source.** Text-to-SQL vLLM SLO — `scripts/start_vllm.sh`, `REPORT.md` §1.

---

## Build the whole pipeline on a cheap, same-model backend before you touch the scarce GPU

**Problem.** The real run was a single booked **1-hour H100** slot. Debugging serving, agent, eval and
dashboards live on that clock is how you waste it.

**Technique.** Point the agent at the *same model* hosted on a cheap OpenAI-compatible endpoint (Nebius
served the identical Qwen3-30B), and build/validate everything there first: agent logic, prompts, eval
harness, tracing, and the dashboard (against a tiny CPU vLLM or a mock `/metrics`). Reserve the H100
purely for the numbers that *must* come from it (SLO latency, final eval).

**When to use.** Any project gated behind a scarce or expensive resource (booked GPU, paid API budget,
prod access). Dev on a representative-but-cheap stand-in; spend the scarce thing on final measurement.

**Pitfall.** "Representative" matters — same model family/version so prompt behaviour carries over. And
it caught a real bug: a schema-render crash that fired on specific DBs would have thrown errors all
through the H100 load test. Found it on the cheap backend, fixed it, walked into the slot clean.

**Source.** Text-to-SQL vLLM SLO — whole build; dev backend = Nebius hosted 30B, final run = H100.

---

## Verify the post-condition of a state-changing call, not the exit code

**Problem.** A control-plane call can return success and still not do what you meant. I scaled a GPU
node group to 0 to stop billing; it exited 0 and the nodes showed `SchedulingDisabled`, so I assumed
they were gone. They were only cordoned, the VMs kept running, and two L40S billed for over an hour
before I caught it.

**Technique.** After any action that mutates external state (scale, delete, deploy, stop), assert the
intended END STATE with a separate read. "Did it return 0" and "did the resource reach the state I
wanted" are different questions.

**When to use.** Every state-changing cloud or tool call where a silent no-op costs money, data, or a
stuck pipeline. Skip it for pure reads.

**Code sketch.**
```bash
nebius mk8s node-group update --id $NG --fixed-node-count 0   # exit 0 != nodes gone
test "$(kubectl get nodes --no-headers | wc -l)" -eq 0 || echo "STILL BILLING: nodes not terminated"
```

**Pitfall.** "Scale to 0" is not "delete". Some managed services cordon-and-drain on scale-down and
never terminate the VM if the drain cannot complete. To truly stop billing, delete the resource and
confirm it returns NotFound.

**Source.** Nebius DDP run. Bill reached the mid-teens of dollars instead of ~$6 because a scale-to-0
left the nodes billing for an hour.

---

## Reference code rots; validate it against the image's real library versions before you pay for the GPU

**Problem.** Known-good reference code, written against older libraries, broke only at runtime inside an
image with unpinned deps. `datasets` 5.0 rejected the bare name `"wikitext"` (it now needs
`Salesforce/wikitext`), and the image had no `openssh-server` that the multi-node runtime expected. On a
paid job, each break is a launch you pay for.

**Technique.** Before the expensive run, exec the exact failure-prone calls inside the real image on CPU:
construct `TrainingArguments`, load the tokenizer, resolve the dataset id. Catch version drift for cents
instead of a GPU-hour.

**When to use.** Any time you run third-party reference code inside an image with unpinned deps, or right
after a major-version bump of transformers, datasets, or torch.

**Code sketch.**
```bash
docker run --rm $IMAGE python -c "
from datasets import load_dataset
load_dataset('Salesforce/wikitext','wikitext-2-v1',split='validation')  # fails fast on a wrong id
"
```

**Pitfall.** `bf16=True` raises on a CPU box ("doesn't support bf16/gpu"), which looks like a real failure
but is just the CPU guard. Use `bf16=False, use_cpu=True` for the offline check and the rest of the API
surface still validates.

**Source.** Nebius DDP run. The first paid launch died at dataset load, the second at SSH setup, both
catchable on CPU beforehand.

---

## A multi-call agent's latency floor is one whole answer; lower the load to find it

**Problem.** My SLO was P95 end-to-end under 5 s at 10 rps. It missed by ~20x (P95 ~108 s). The obvious
read is "too much load" and the obvious fix is to shed load. Both are wrong when the agent makes several
sequential LLM calls per answer.

**Technique.** One agent run is N sequential calls (generate -> verify -> revise = 2-3), so its latency
floor is N × per-call latency, not one call. To find the floor, run the load test *down* until the
backend is idle and watch the median. I swept 10 -> 5 -> 2 rps. At 2 rps the GPU was idle enough that
only 1 of 360 requests timed out, yet P50 was still 5.65 s. The median answer alone breaks a 5 s budget,
so no rps reduction can ever meet it; the fix is agent-side (fewer calls, or async so calls overlap) or a
faster model, not a serving flag. Back-of-envelope: sustainable rps ≈ backend_req/s ÷ calls_per_run
(~15 ÷ ~2.5 ≈ 6), so 10 rps was always going to queue without bound.

**When to use.** Any latency SLO on a multi-step agent. Run the load test DOWN as well as up.

**Code sketch.**
```bash
for rps in 10 5 2; do
  python load_test/driver.py --rps $rps --duration 180 --out lt_$rps.json   # P50 at the bottom rung = the floor
done
```

**Pitfall.** A stable run is not a passing run. At 2 rps the dashboard looked calm (no timeouts, KV idle),
which tempts "it's fine at low load" — but the median was already over budget. Read the percentile, not
the vibe.

**Source.** Text-to-SQL vLLM SLO — `load_test/driver.py`, `REPORT.md` §3 (the 10/5/2 rps sweep).

---

## A targeted metric can move while the SLO doesn't — confirm the end-to-end followed

**Problem.** Phase 6 asks you to "change one thing, confirm the metric moved." I dropped `--enforce-eager`
to turn CUDA graphs on, betting decode speed was the wall. TPOT and P50 improved (41 s -> 37 s). I almost
called it a win. End-to-end P95 did not budge (108 s -> 116 s, inside run-length noise).

**Technique.** When you tune a lever, verify the *SLO* metric followed, not just the metric you targeted.
A targeted metric improving while the SLO stays put is a real result: it proves the binding constraint is
elsewhere. Here it localized the wall to throughput / agent shape, not decode.

**When to use.** Every "one lever, re-measure" iteration where a metric and the SLO are not the same number.

**Pitfall.** The trap is stopping at "the metric I aimed at moved." That is necessary, not sufficient.
Re-check the actual end-to-end SLO every time before you claim the change helped.

**Source.** Text-to-SQL vLLM SLO — baseline (enforce-eager) vs CUDA-graphs run, `REPORT.md` §3.

---

## Read the dashboard skeptically: threshold lines aren't data, coarse histogram tails lie

**Problem.** I misread my own Grafana board twice and started building a diagnosis on it. (1) The KV-cache
panel had threshold lines at 0.8 and 0.95; with real usage near 5% (a flat line on the floor) the two
markers dominated the view and I read "KV pinned at 100%." (2) The e2e-latency panel showed ~8-minute P99
spikes that were `histogram_quantile` artifacts: vLLM's top buckets are very wide, so when load stops and
the tail goes sparse the quantile lands in one giant bucket and reads minutes.

**Technique.** Treat threshold/marker lines as not-data (render them dashed, label "actual vs limit"), and
distrust tail percentiles from coarse histograms — cross-check against a second signal (TTFT, TPOT,
server-side e2e, the raw counter) before acting. Real per-call latency here was ~6-7 s, nothing like
8 minutes.

**When to use.** Any time a "pinned" gauge or a tail percentile drives a diagnosis.

**Pitfall.** The human caught both misreads before I committed to them. Say the raw observation out loud
("KV looks at 100%, P99 reads 8 min") so a second pair of eyes can sanity-check it; a confident wrong read
of the dashboard sends the whole diagnosis down the wrong path, and the panels that mislead you will
mislead the next reader too (so fix the panel, not just your notes).

**Source.** Text-to-SQL vLLM SLO — `serving.json` (KV + e2e panel descriptions added after the misread).
