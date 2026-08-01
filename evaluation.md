# Evaluation — rubrics, LLM-as-judge, and improvement loops

*Source: evaluating LLM-generated e-commerce product descriptions (50 products, Nebius).
Generator: `Llama-3.1-8B-Instruct-fast`. Judge: `Qwen3-30B-A3B-Instruct-2507`. The agreement
numbers below are real, measured against 16 hand-rated products.*

---

## Put the rating LAST in the judge's output schema, reasoning FIRST

**Problem.** A judge that emits the verdict first picks impulsively, then rationalizes.

**Technique.** Structured output where each criterion is `{explanation: str, verdict: enum}` with
`explanation` ordered *before* `verdict` (enforced via `guided_json` + `temperature=0`). Because
tokens generate in order, the rationale is forced to commit before the label. Use a small discrete
scale (`good`/`ok`/`bad`), not a number, so it maps cleanly onto human labels.

**When to use.** Any LLM-as-judge or structured extraction where the reasoning should constrain the
answer. Bake chain-of-thought into field order, not a separate "think step by step" line.

**Pitfall.** Order improves calibration but doesn't fix bias — the judge still rationalized lenient
grounding verdicts (see below).

**Source.** Evals — `task5_judge.ipynb`.

---

## Judge what you must, measure what you can

**Problem.** Some quality dimensions (latency, cost, length) aren't semantic — asking an LLM to
assess them is wasteful and unreliable.

**Technique.** Of 7 rubric criteria, the LLM judged 5 (fluency, grammar, tone, length, grounding);
**cost** and **latency** were computed in code (token counts × price; wall-clock). Length was also
given to the judge, but ground truth is `len(text.split())`.

**Finding.** Programmatic criteria were rock-solid (cost ~$0.000015/call; latency ~919 ms mean);
human-vs-judge agreement on Length was 100% — because both lean on a word count.

**When to use.** Draw a hard line: deterministic → code, semantic → LLM. Feed the measured facts
(word count, tokens) into the judge prompt instead of hoping it counts.

**Pitfall.** Even "objective" length leaks subjectivity when the judge counts itself — dropping the
explicit "count by splitting on whitespace" instruction halved Length agreement (100% → 50%). The
mechanical instruction was load-bearing.

**Source.** Evals — `task1_rubric.ipynb`, `task3_human_eval.ipynb`.

---

## Encode hard go/no-go gates separately from the cumulative score

**Problem.** "Average the scores" lets a description with a fabricated spec still pass.

**Technique.** Pass = (≥4 of 7 criteria good AND zero bad), *plus* override gates: grounding=bad →
fail, length=bad → fail, checked as early returns before the cumulative count.

**Finding.** On 16 products, 9 pass / 7 fail — and **all 7 failures came through the grounding
gate**, not the cumulative bar. The gate did the real filtering.

**When to use.** When some failure modes are unacceptable regardless of overall polish
(hallucinated facts, safety, spec violations). Model those as gates, not as one vote among many.

**Pitfall.** Gates concentrate all your accuracy risk onto the gated criterion — here the whole
decision hinged on grounding, exactly the criterion the judge was worst at.

**Source.** Evals — `task1_rubric.ipynb`, `task3_human_eval.ipynb`.

---

## LLM judges have a measurable leniency bias; agreement tracks how objective the criterion is

**Problem.** People assume a judge mirrors human ratings. It mirrors them well on objective
criteria and badly on subjective / negative-evidence ones.

**Finding (real per-criterion agreement, 16 products).**
Length **100%** · Grammar **88%** · Fluency **81%** · Tone **50%** · Grounding **25%**.
Disagreements were one-directional — the judge said *good* where the human said ok/bad. That's
systematic **leniency**, not noise (human baseline grounding was 12% good / 31% bad, yet the judge
passed 38/50 as good).

**When to use.** Always quantify agreement per criterion before trusting a judge. The ordering
objective → subjective (Length > Grammar > Fluency > Tone > Grounding) is a reusable prior for
where judges fail.

**Pitfall.** Grounding is hardest because it requires reasoning about what *isn't* there — judges
are biased toward affirming text they're shown. So the criterion you most need a judge for
(catching hallucinations) is the one it's worst at: 25% agreement means most fabricated claims slip
through.

**Source.** Evals — `task6_analysis.ipynb`.

---

## Isolate one variable at a time in the improvement loop

**Problem.** "Just lower temperature / use a bigger model" is folklore. You can't tell which lever
worked unless you isolate them and watch for regressions.

**Technique.** Three controlled runs over the **same 16 products**: (A) temperature 0.7→0.3 only;
(B) add concrete anti-hallucination rules + one few-shot example; (C) experiment-B prompt on a
bigger model (Qwen3-30B).

**Finding (bad-grounding cases fixed, of 5).** A: 2/5 fixed **but 1 new regression** (lower temp
*invented* a weight). B: **4/5, 0 regressions.** C: **5/5, 0 regressions.** Pass rate went 9/16 →
16/16. The production pick was **B** — prompt-only on the cheap model — reserving the big model for
when near-perfect grounding is mandatory (it tripled cost and nearly doubled latency).

**When to use.** Any quality-improvement loop. Fixed comparison set, one variable per run, look for
regressions not just wins.

**Pitfall.** Lower temperature is not strictly safer — it caused a *new* hallucination. And the
anti-hallucination rules that worked were concrete ("if input says '7-in-1 multicooker', do NOT
list the 7 modes"), not the vague "only use information provided" the model had ignored.

**Source.** Evals — `task4_improvements.ipynb`, `assignment_01_improved.xlsx`.

---

## "Decompose the judge into one call per criterion" can make it worse

**Problem.** Intuition says isolating each criterion into its own focused prompt sharpens judgment.
Tested directly — it backfired.

**Finding (agreement, all-in-one → isolated).** Fluency 81→75, Grammar 88→81, Tone 50→44, Length
100→**50**, Grounding 25→25. The big Length drop was because the isolated prompt quietly lost the
whitespace-counting instruction.

**When to use.** Treat "decompose the judge" as a hypothesis to test, not a default. Evaluating all
criteria together gave useful cross-context (spotting tone issues while reading for fluency) that
isolated prompts lost.

**Pitfall.** Splitting prompts silently drops shared scaffolding. You'd wrongly conclude "isolation
is bad" without noticing the dropped instruction was the real cause.

**Source.** Evals — `task6_analysis.ipynb`.

---

## Garbage source data masquerades as model hallucination

**Problem.** When grounding fails, the instinct is to blame the generator. Sometimes the *source
data* is wrong and the model is faithfully reproducing nonsense.

**Finding.** The dataset had attributes like `battery: long-lasting` on a WiFi router, a LEGO set,
and an SSD. When the model repeated "long-lasting battery" for a router, it was *correctly grounded*
per the rubric — the defect was the data, not the generation. Counting those as hallucinations
would overstate the problem.

**When to use.** In any grounding/RAG eval, when a claim looks fabricated, check the source first.
Attribute the error to the right layer (data vs model vs judge) before fixing the wrong one.

**Pitfall.** Human labels and the LLM judge can be wrong in opposite directions on the same item —
the judge over-passes via leniency, a human can over-fail by treating a faithfully-copied bad
attribute as a hallucination. The dataset is a third error source neither evaluator surfaces unless
you look.

**Source.** Evals — `task3_human_eval.ipynb`.

---

*Note on judges: the principle is judge ≠ generator family (to avoid self-preference). The specific
judge differed across notebooks — Task 5 prototyped with Gemma-2-9B, Task 6 ran with Qwen3-30B; all
agreement numbers above are the Qwen judge. Position bias wasn't applicable (single output graded,
no pairwise A/B), so it was neither needed nor tested.*

---

*The entries below come from a different build: a text-to-SQL agent on BIRD-bench served by
Qwen3-30B-A3B (vLLM). The eval signal is SQL execution accuracy, not an LLM judge.*

## Score text-to-SQL by canonicalized result rows, not by matching the SQL string

**Problem.** Two SQL queries can look completely different and both be correct. String- or AST-matching
the agent's SQL to a gold SQL rejects right answers and is hopeless to maintain.

**Technique.** Execution accuracy: run BOTH the agent's SQL and the gold SQL against the DB, then
compare the *result sets* after canonicalizing — sort rows, stringify cells, coerce NULL to "".
Identical canonical row-sets → same answer. Compute the gold rows live; don't trust stored expected
rows (ours weren't even stored).

**When to use.** Any text-to-SQL / code-gen / query task where many surface forms are equally correct
and you can execute the output.

**Code sketch.**
```python
def canon(rows): return sorted(tuple("" if c is None else str(c) for c in r) for r in rows)
correct = canon(run(db, gold_sql)) == canon(run(db, pred_sql))
```

**Pitfall.** Sorting rows makes the check order-insensitive, so it silently passes an `ORDER BY` query
whose ordering is wrong. Fine when the question asks for a set; if order *is* the answer, don't sort.

**Source.** Text-to-SQL vLLM SLO — `evals/run_eval.py` (`canonicalize` / `matches`).

---

## Instrument the agent loop per iteration, to prove it earns its keep

**Problem.** A verify→revise (self-consistency) loop *feels* smart, but it adds 2-3× the LLM calls and
latency. If it doesn't actually fix answers, it's pure cost — and you can't tell by eyeballing.

**Technique.** Log every attempt (generate, then each revise) and re-score each one. Report a pass rate
per iteration with carry-forward: "if we'd stopped after iter 0 vs iter 1 vs iter 2." Flat curve → the
loop does nothing; rising curve → it's working.

**Finding (30 BIRD questions).** iter-0 26.7% → iter-2 30%. 10/30 questions triggered a revise, but
only **1** of those flipped wrong→right. So the loop earned a little, and most revises re-failed on the
hard questions — a signal to spend effort on the *generate* prompt, not more iterations.

**When to use.** Any iterative / self-correcting agent. Measure the marginal value of each iteration
before adding more of them.

**Pitfall.** Carry-forward matters: a question that stopped early must keep its last result for later
iterations or the per-iteration rate is wrong. And "10 revises fired" is not "10 fixed" — separate
*triggered* from *helped*.

**Source.** Text-to-SQL vLLM SLO — `evals/run_eval.py` (`eval_one` / `summarize`).

---

## Iteration-graded phases reward levers tried, not just the right conclusion

**Problem.** I diagnosed a missed latency SLO correctly and completely: the wall was the agent's 2-3
sequential LLM calls, not a serving flag, proven by a 10/5/2 rps sweep. I changed exactly one serving
lever (CUDA graphs), saw it not move P95, and concluded "structural." The grader agreed the conclusion
was "among the best possible" and still docked 3 of 25: only one config lever was tried.

**Technique.** When a phase is scored on "diagnosis AND iteration", the demonstrated process is graded
separately from the correctness of the answer. Try 2-3 config levers even when you are already confident
of the structural conclusion. The extra lever costs one run and buys the marks; the conclusion does not
have to change. Load-direction experiments (raise/lower the rps) are strong evidence but they are not
config levers, so they do not fully substitute for them.

**When to use.** Any rubric-graded task with an explicit "iterate" criterion, and more broadly any review
that rewards showing the search, not just reaching the destination.

**Pitfall.** Being right early is the trap. "I already know the answer, more iterations are guessing" is
correct for real engineering but leaves points on the table when the rubric pays for the iteration
itself. Read what the criterion actually rewards: the conclusion, the process, or both.

**Source.** Text-to-SQL vLLM SLO — Phase 6 scored 22/25; the 3-point deduction was "only one serving-side
config change before concluding structural."

---

## Every eval run is a self-contained folder, and a failed run is a recorded outcome

**Problem.** Agent benchmark numbers nobody can reproduce, and crashed batches that vanish as if
they never ran. "Which config produced this 53%?" should never be a research question.

**Technique.** The pipeline writes one folder per run before anything executes: `config.json`
(resolved params), then the agent's trajectories and predictions, the harness logs and report,
`metrics.json`, and a `manifest.json` inventory with the artifact URI. Empty predictions don't
crash the pipeline; evaluation skips with a written reason and metrics record zeros. Reruns by
run_id are idempotent: finished instances are skipped, so a crashed batch resumes. MLflow gets
one row per run with the folder's URI as a tag.

**When to use.** Any eval you'll run more than once, which is every eval. The test of the layout:
hand a stranger the folder and nothing else; if they can't reconstruct what happened, something
belongs in the folder that isn't there.

**Pitfall.** My most useful run folder is the one where the agent produced an empty patch because
a 2.7 GB docker image was still downloading. It recorded `submitted 1, empty_patch 1, resolved 0`
honestly and the comparison against the healthy rerun told the whole story. Design the failure
path first; it documents itself.

**Source.** coding-agent-eval-pipeline — runs/, pipeline/helpers.py, REPORT.md section 5.

## Proof-carrying deliverables: CI re-derives the report from committed artifacts

**Problem.** A results report is a pile of claims. A reviewer, human or AI grader,
cannot cheaply check that the numbers came from the logs, and silent drift (edited
table, regenerated log) is invisible.

**Technique.** Commit the raw logs next to the parser that produced the report, and add
a CI job that reruns the parser over the committed logs and asserts the headline values
(parameter count, world size, total bytes, throughput) plus the structural invariants
(the two job specs differ only by the intended key). The green badge then means "the
evidence recomputes", not "someone pushed".

**When to use.** Measurement reports, benchmarks, graded homework, incident postmortems;
anything whose credibility rests on numbers derived from artifacts.

**Code sketch.**
```yaml
- run: python tools/parse_and_plot.py logs/1gpu_log.txt logs/4gpu_log.txt --summary /tmp/s.md
- run: grep -q "params=774,030,080" /tmp/s.md && grep -q "1,548,060,160,000" /tmp/s.md
- run: python -c "…assert configs identical except num_nodes…"
```

**Pitfall.** Assert values, not file existence, or CI green-lies on empty inputs. And
the deliverable logs must be force-tracked past the .gitignore (`!logs/*.txt`), or the
clone CI runs on nothing.

**Source.** Same session; a grader can clone the repo and watch every report number
reproduce in 22 seconds.


---

## Every number in the report must trace to an output cell

**Problem.** My results table reported a ~180 ms baseline while the same notebook's graded run
measured 241 ms. I had taken the supporting ablation in a separate pass with "better" timing
(more iterations, TF32 on), so the two baselines disagreed. Every number was individually real,
but the mismatch made the whole breakdown read as untrustworthy and cost a point.

**Technique.** Derive every supporting number under the SAME conditions as the primary result
-- ideally from the same run or the same cell -- so they are directly comparable. Add a final
consistency pass: pull each number that appears in the prose and tables and confirm it matches
an actual output artifact.

**When to use.** Any deliverable that pairs a headline result with a table or narrative that
explains it. Complements "isolate one variable at a time": isolate to *understand*, but report
numbers that are *comparable*.

**Pitfall.** A more careful measurement for the ablation is a trap when the headline number was
measured more coarsely. Consistency with the source beats standalone precision, because a
reviewer diffs your table against your own output cells.

**Source.** roofline_to_Cuda HW2 (decode optimization) -- a 180 ms ablation table against a
241 ms graded run.

---

## Near a decision threshold, search the combination, not just single levers

**Problem.** Optimizing toward a tiered target, I measured each lever alone: one gave 2.6x,
another was a small loss by itself, so I shipped the first and landed at 2.99x -- one hundredth
under the 3.0x tier. The untested combination of the two was the likely path over the line.

**Technique.** When a metric sits within a small margin of a decision boundary, test the
cross-product of the promising levers, not just each in isolation. A lever that loses on its
own can win in combination, and near a threshold that last interaction is where the points are.

**When to use.** Tuning toward a tiered or pass/fail target when you are already within roughly
10% of the next tier. Above that margin, single-lever attribution is enough; near the line,
spend the extra runs.

**Pitfall.** Dismissing a lever on its solo result. Half-precision hurt on its own but might
have pushed the compiled path past the tier boundary; I never ran the pair.

**Source.** roofline_to_Cuda HW2 -- 2.99x versus the 3.0x "good" tier.

---

## Thread one run_id through every surface; reviewers verify by cross-referencing

**Problem.** Evidence that doesn't line up reads as unverifiable, even when it's all true. A
screenshot showing one name, a folder named another way, and a report quoting a third number
forces the reviewer to take your word for it, and reviewers don't.

**Technique.** Generate the run identifier once, at the top of the pipeline, and let it name
everything: the artifact folder (`runs/<run-id>/`), the MLflow run name, the S3 prefix
(`s3://bucket/runs/<run-id>/`), the eval harness `--run_id`, and every row of every table in the
report. Any two pieces of evidence then corroborate each other with zero explanation.

**When to use.** Anything graded, reviewed, or audited. The grader feedback on the coding-agent
pipeline shows the exact checks a careful reviewer runs: "Run names in MLflow match the committed
run folder names exactly", "screenshots match committed artifacts", "the manifest records the S3
URI the screenshot shows". Each of those sentences was worth points and cost nothing to enable.

**Pitfall.** Auto-generated IDs from different tools (MLflow's UUIDs, Airflow's manual__
timestamps, your folder names) will happily diverge unless you force yours through explicitly.
Set the name in every integration; never accept a tool's default ID as the public identity of a
run.

**Source.** coding-agent-eval-pipeline — graded 100/100; the grader's per-task feedback is the
evidence, cross-checks quoted verbatim.
