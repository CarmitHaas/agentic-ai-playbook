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

---

## Set an explicit max_tokens on agent loops; provider defaults kill episodes silently

**Problem.** My first mini-swe-agent episodes all died with `RepeatedFormatError`: the model hit an
output-token ceiling mid-reasoning (`finish_reason=length`) before emitting the tool call the
harness expects, three times in a row, episode over. The instructor's own sample batches show the
same thing: 3 of his 4 runs ended with every instance in that state. Nobody set a limit; the
provider's default applied, and the default was too small for a thinking model.

**Technique.** Always pass an explicit output budget on agentic loops and expose it as a run
parameter, not a constant. In mini-swe-agent that's one config override the pipeline injects on
every run: `-c model.model_kwargs.max_tokens=<budget>`.

**When to use.** Any harness where the model must finish with a parseable action (tool call, bash
block, JSON). Truncation doesn't error, it just produces garbage the loop can't parse.

**Pitfall.** The exact number mattered less than setting one: 4096 and 8192 both worked. My count
across the project: no explicit limit, 0 of 11 episodes submitted a patch; explicit limit, 19 of 19.

**Source.** coding-agent-eval-pipeline — REPORT.md section 6.

---

## Orchestrate at the right altitude: harness inside the experiment, workflow engine around it

**Problem.** "Airflow runs LLM steps, LangGraph orchestrates steps, so why not LangGraph for the
pipeline?" Because there are two loops at different altitudes and each tool owns one.

**Technique.** The inner loop (think, act, observe; seconds per step; state in memory) belongs to
the agent harness: LangGraph, or mini-swe-agent's hand-rolled equivalent. The outer loop (batch
jobs, retries on infrastructure failures, run history, multi-user triggering; minutes to hours)
belongs to a workflow engine: Airflow. Same split on the observability side: Langfuse-style
tracing looks inside one conversation; MLflow compares whole runs. Four tools, four altitudes,
no overlap when placed right.

**When to use.** Any time an agent experiment graduates from "I run a script" to "the team runs
experiments". Putting Airflow inside the agent loop adds seconds of scheduler latency per model
turn; putting LangGraph around batch jobs means rebuilding retries, logs, history, and UI.

**Pitfall.** The tempting shortcut is one tool for everything. The tell that you got the altitude
wrong: either your DAG has a task per model turn, or your agent framework has a cron wrapper.

**Source.** coding-agent-eval-pipeline — evaluate_agent DAG over mini-swe-agent episodes.

---

## One step entrypoint, two isolation levels

**Problem.** Pipelines developed locally as subprocesses get rewritten for containers at deploy
time, and the rewrite is where bugs breed.

**Technique.** Every pipeline step is `python -m pipeline.run_step <step> <run_dir>`. The DAG
decides isolation per environment: local mode wraps it in a subprocess, docker mode passes the
identical command to DockerOperator. The entrypoint rebases paths from the run folder's own
location, so a run created on a host works inside a container mounted elsewhere. I deployed the
locally-tested pipeline to the VM unchanged; the process boundary I drew on day one became the
container boundary on day two.

**When to use.** Any DAG you develop under `airflow standalone` and ship under compose or k8s.
The step-entrypoint list is also exactly the interface a later KubernetesPodOperator needs.

**Pitfall.** Keep the heavy dependencies (mlflow, boto3) out of the orchestrator's environment;
the entrypoint owns them. Airflow's env stays clean and the migration path stays open.

**Source.** coding-agent-eval-pipeline — pipeline/run_step.py, dags/evaluate_agent.py.

## Read the operation error body, not the console status label

**Problem.** Cloud consoles compress machine errors into two-word labels. gRPC code 8,
`ResourceExhausted`, covers both "the datacenter is out of hardware" and "your quota
blocked this", and the console renders both as "Resource exhausted". I spent ~90 minutes
chasing GPU capacity across three pools (on-demand, preemptible, even a whole 8-GPU host)
while the real error was a public-IPv4 quota: limit 3, my 4th node requested the 4th.

**Technique.** On any provisioning or state-changing API failure, the first move is
fetching the operation record and reading `status.message` and `status.details`. The
structured details name the violated quota with limit and requested, and `retry_type`
says whether the platform will ever retry it for you (`NOTHING` means your retry loop is
decorative).

**When to use.** Every cloud allocation failure, and any time two signals disagree; the
capacity advisor kept saying HIGH while allocations failed, and that disagreement itself
was the message that the label was lying.

**Code sketch.**
```bash
nebius compute instance list-operations-by-parent --parent-id <project> --format json \
  | jq '.operations[] | select(.status.code != 0) | {desc: .description, msg: .status.message, retry: .status.details[0].retry_type}'
# AWS analog: CloudTrail event / describe-instances StateReason, not the console badge
```

**Pitfall.** The pre-flight version of the same mistake: audit quota for EVERY resource
the template allocates times node count (instances, disks, and above all public IPs),
not just the marquee GPU number. `4 nodes x 1 public IP = 4 > 3` was knowable before
creating anything.

**Source.** DDP scaling anatomy (4x H100 on Nebius mk8s, 2026-08). Five failed
allocations across three pools; fixed in ten minutes by a no-public-IP node group once
the operation body was read. Extracted live into the `triaging-cloud-allocation-failures`
skill.

## Napkin-math the communication budget before the first GPU-hour

**Problem.** Distributed training experiments get launched on vibes, then the team burns
paid GPU-hours debugging "why is 4x hardware slower than 1x" as if it were a bug. In my
course cohort, multiple people hit the same 90%-communication wall days apart, treated it
as a misconfiguration, and some pivoted topology entirely to make the numbers look right.

**Technique.** Ten minutes of arithmetic before provisioning: gradient payload =
params x bytes per grad; ring all-reduce moves 2(N-1)/N x payload per GPU per step;
divide by the realistic interconnect bandwidth (TCP ~5-10 Gbit/s, InfiniBand 100x that)
and compare against the estimated compute per step (FLOPs / achievable TFLOPS). Then
pre-register the falsifiable check: write down which log line decides between your two
worlds (here: `via NET/Socket` vs `via NET/IB`) before the run starts.

**When to use.** Any multi-device or multi-node training plan, before committing money;
any benchmark where "slow" could be either a bug or the phenomenon itself.

**Code sketch.**
```python
params = 774_030_080
payload = params * 4                     # fp32 grads: 3.10 GB
ring    = 2 * (4-1)/4 * payload          # 4.64 GB per GPU per step
tcp     = ring / (6e9/8)                 # ~6 Gbit/s TCP -> ~6.2 s ceiling
compute = 6 * params * 16*512 / 400e12   # batch tokens / 400 TFLOPS -> ~0.1 s
# comm >> compute: the experiment is communication-bound BEFORE you rent anything
```

**Pitfall.** An envelope estimate that says "comm-bound" is not a reason to cancel; it
is the hypothesis that turns the run into a measurement instead of a surprise. And when
the assignment (or the client ask) is "explain the scaling", the underperformance IS the
deliverable; fixing the topology to prettier numbers answers a question nobody asked.

**Source.** DDP scaling anatomy: predicted comm-bound over TCP in cold prep, measured
4.303 s of comm inside a 4.600 s step the same day. Cohort peers hit the identical wall
unwarned days apart.


---

## Probe for silent degradation before the metered run

**Problem.** A freshly-provisioned GPU box can be quietly wrong in a way that never raises an
error. `pip install torch==2.12.1` pulled the newest CUDA build (`+cu130`); the rented H100's
driver only spoke CUDA 12.4, so `torch.cuda.is_available()` returned False with just a warning.
The graded notebooks would have run to completion on CPU and written plausible placeholder
numbers — on a GPU I was paying for by the minute.

**Technique.** Before the real run, run a ten-second probe that fails LOUDLY on the exact
silent-degradation conditions: is the accelerator actually visible, is it the device you
rented, and does the expensive path (here `torch.compile`) really engage instead of falling
back. Start the graded run only after the probe prints the expected device and a compiled
result.

**When to use.** Any one-shot, metered, or graded run on hardware or an environment you did
not build yourself and cannot cheaply redo.

**Code sketch.**
```python
import torch
assert torch.cuda.is_available(), "CUDA missing -- would silently run on CPU"
print(torch.cuda.get_device_name(0))                   # the card you actually rented?
f = torch.compile(lambda x: x * 2 + 1)
print(float(f(torch.randn(8, device="cuda")).sum()))   # does compile really fire?
```

**Pitfall.** `pip install torch==X` grabs the newest CUDA build, which an older driver cannot
run. Match the build to the driver: a `cu126` wheel runs on a 12.4 driver via CUDA
minor-version compatibility; a `cu130` (major 13) wheel does not. The failure mode is silence,
not a crash.

**Source.** roofline_to_Cuda (H100 perf homework) -- the probe caught cu130-vs-driver-550
before the graded run, saving a full CPU-garbage run on a paid H100.
