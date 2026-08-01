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

*Source for the two entries below: a multimodal assignment - BLIP captioning + VQA and CLIP face recognition, run locally on CPU. Numbers are real, from the graded notebook.*

## Pick evaluation data that is allowed to fail

**Problem.** An eval set where the model scores 100% teaches you nothing about its limits. My first identity set for a face-recognition task (five early-2000s politicians) scored 15/15, which looked great and hid every weakness.

**Technique.** Choose eval items near the decision boundary on purpose. I swapped in five entertainers with two genuine look-alikes (Winona Ryder, Angelina Jolie) and added one out-of-set person (Tom Cruise) who belongs to no known class. Accuracy dropped to 13/15 and the failures became the useful part of the writeup.

**When to use.** Any time a first eval pass looks suspiciously perfect. A ceiling result usually means the test is too easy, not that the system is done.

**Finding.** Politician set: 15/15, nothing to analyze. Celebrity set: 13/15, with a clean Winona/Jolie confusion (sim 0.90 and 0.91) and a stranger forced onto the nearest known face.

**Pitfall.** Do not manufacture a failure by breaking the system. Manufacture it by choosing harder, fairer data. The look-alikes and the stranger are legitimate inputs, not sabotage.

**Source.** Multimodal task - CLIP face recognition on LFW.

---

## Write the analysis from captured outputs, not the outputs you expect

**Problem.** It is tempting to write "the model probably says X" and move on. On greedy decoding the model is deterministic, so guessing is pure risk for no reason.

**Technique.** Run the real pipeline headless first, capture every prediction to a file, then write the notebook analysis by quoting those exact strings and numbers. The graded run reproduces them because decoding is deterministic (temperature 0 / greedy).

**When to use.** Any deliverable that cites model output. The preview equals the final run, so the analysis is correct before the reader ever executes a cell.

**Finding.** Every quoted caption, VQA answer, similarity score, and accuracy in the writeup matched the graded run exactly, because I authored them from a captured run rather than from memory.

**Pitfall.** This only holds while decoding is deterministic. Turn on sampling and the captured run and the reader's run diverge, so the quoted numbers go stale.

**Source.** Multimodal task - BLIP captioning + VQA.

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

---

## Compare runs at the point where they are still comparable

**Problem.** You sweep a hyperparameter, then read the metric at the end of every run and rank the
settings. But by the end, the runs are no longer one system under different settings, they are
different systems. Whatever the parameter did early on has been amplified into a different state,
and the metric is now reporting the state, not the parameter.

**Technique.** Reset to identical weights and reseed the RNG before every run, so every run sees
the byte-identical first batch. Then read the cross-run comparison at **iteration 0**, where the
only difference is the knob, and read the end-of-run numbers as outcomes rather than as a
measurement of the knob. Report both.

**When to use.** Any sweep over a system that accumulates state: RL runs, fine-tuning configs,
agent memory settings compared after N turns, retrieval configs compared after index drift.

**Code sketch.**
```python
def train_ex(c, name, ...):
    torch.manual_seed(EX_SEED)
    policy.load_state_dict(copy.deepcopy(policy_fresh_state))  # train() mutates the GLOBAL policy
    ...
    H["adv_std"].append(adv.std().item())

# the honest comparison is index 0: identical weights, identical rollout, one dial different
[runs[n]["adv_std"][0] for n in names]   # 0.1241, 0.1951, 0.2610  monotone in lambda
[summ(runs[n])["adv_std"] for n in names] # 0.090, 2.036, 0.452   ordering destroyed
```

**Pitfall.** This bit three times in one session and inverted the conclusion every time. Worst case:
a converged policy and a leashed policy both show tiny importance-ratio drift, for opposite reasons.
The end-of-run table said clip=0.05 moved the policy *most*; at iteration 0 it moved it *least*,
which is the true effect of a narrow clip. Also: check whether the training function mutates global
state. Two configs run back to back without a reset are not a comparison, the second inherits the
first.

**Source.** Glass-box PPO lab (Nebius Academy session 2), exercises 1, 2 and 4.

---

## A metric whose definition contains the knob you are turning is not a measurement

**Problem.** PPO's `clipfrac` is defined as `|ratio - 1| > clip_eps`. Sweeping `clip_eps` and
reporting `clipfrac` changes the ruler and the thing being measured at the same time. Narrowing the
clip from 0.2 to 0.05 moved `clipfrac` by about 100x (0.003 to 0.291) while the policy's actual
step size changed by well under 2x.

**Technique.** For every threshold-based metric, log a threshold-free twin and lead with it. Keep
the threshold metric for continuity with everyone else's dashboards, but do not draw conclusions
from it across settings of its own threshold.

**When to use.** Any tuned threshold that also appears in a reported metric: pass-rate at a judge
score cutoff while tuning the cutoff, "responses under budget B" while tuning B, retrieval hit-rate
at k while tuning k.

**Code sketch.**
```python
st["clipfrac"].append(((ratio - 1).abs() > c.clip_eps).float().mean().item())  # ruler moves with the knob
st["ratio_drift"].append((ratio - 1).abs().mean().item())                      # same ruler everywhere
```

**Pitfall.** The threshold-free twin is not automatically comparable either. `ratio_drift` still has
to be read at the shared starting state (see the entry above).

**Source.** Glass-box PPO lab, exercise 1. A grader called this out as the strongest single idea in
the submission.

---

## Find out what dominates a metric before you use it as a proxy

**Problem.** I logged total gradient norm as a variance meter for the policy gradient. It read 15.6,
221.1 and 246.4 across three runs, which looked like a clean signal about update variance. It was
not. The loss also contains `vf_coef * value_loss`, and the value head was regressing a return that
grew from 0.1 to 14 during the run, so the gradient norm was mostly reporting the critic's
regression error, ordered by how much reward each run had found.

**Technique.** Before treating an aggregate as a proxy for X, decompose it and ask what term is
largest. If you cannot decompose it cheaply, run the ablation that removes the suspected dominant
term and see how much of the metric goes with it.

**When to use.** Any composite metric: total loss, total latency, total token spend, "score".

**Code sketch.**
```python
gn = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
# same policy learning, vf_coef 0.1 -> 0.0:  grad_norm 221.1 -> 8.7
# 96% of the gradient was the critic catching up, none of it was doing the learning
```

**Pitfall.** The confirming ablation arrived two exercises later, so three intermediate conclusions
had already been written against the bad proxy. State the caveat where it is discovered *and* at the
top of every later section that leans on the metric, not only where you found it.

**Source.** Glass-box PPO lab, exercises 1 and 5. The grader flagged the late placement of this
caveat as the main thing to improve.

---

## A penalty coefficient is only meaningful as a share of the objective

**Problem.** The same KL coefficient beta = 0.05 was a healthy leash on one reward function
(`beta*KL/R` = 11.6%, reward climbed to 13.7) and a strangling one on another (34.1%, reward stuck
at 1.5). Nothing about the coefficient changed. The reward's achievable scale did.

**Technique.** Report and tune the dimensionless share, `penalty_coefficient * penalty_magnitude /
objective_magnitude`, not the raw coefficient. That share is what transfers between tasks, reward
functions and datasets. Here every regime that behaved sat between 4% and 12%.

**When to use.** KL penalties in RLHF, DPO's beta, regularisation weights, any auxiliary-loss
coefficient, and cost-vs-quality tradeoff weights in an agent's scoring function.

**Code sketch.**
```python
share = beta * kl_per_seq / reward
#  0.05 on positive-word reward -> 11.6%   healthy
#  0.05 on '!'-count reward     -> 34.1%   policy strangled, never learns
#  0.005 on '!'-count reward    ->  3.9%   KL blows out to 46 and the text degenerates
```

**Pitfall, and the one the reviewer marked down.** Do not calibrate the coefficient from a run that
the coefficient itself suppressed. I set the new beta from the reward the throttled run achieved,
which is circular, and it overshot. Iterating fixed it, but a small log-spaced sweep (5 values from
0.005 to 0.1) is the systematic answer and does not depend on the previous run's endpoint.

**Source.** Glass-box PPO lab, exercise 3.

---

## Commit the prediction and its falsifier in writing before the run

**Problem.** Reading results and then explaining them always feels like understanding. It is not
distinguishable, after the fact, from fitting a story to whatever came out.

**Technique.** Before each run, write numbered, specific, falsifiable predictions plus one line
naming what result would prove the mechanism wrong. Run. Then write the verdict against the
predictions you actually made, keeping the failures visible and diagnosing them.

**When to use.** Any parameter study or ablation, and any agent change you are about to defend to
someone. It costs about five minutes per experiment and it is the only thing that makes a wrong
prediction more valuable than a right one.

**Code sketch.**
```markdown
**What I predict, before running.**
3. Reward: eps=0.6 fastest, with the highest KL and the noisiest curve.

**What would falsify me:** if ratio drift is flat across all three, the clip is not the binding
constraint and gradient-norm clipping is doing the work.
```

**Pitfall.** The falsifier is the part people skip, and it is the part that pays. In this lab the
falsifier fired: drift was nearly flat, and the real trust region turned out to be
`clip_grad_norm_(params, 1.0)`, which was scaling the update down by a factor of about 220. Without
the pre-committed falsifier that reads as a failed experiment instead of the finding it was.

**Source.** Glass-box PPO lab, all five exercises. Graded 100/100, with the protocol itself cited
as the reason.

---

## When a locked harness computes a number wrong, recover it from the raw artifact and say where it came from

**Problem.** The graded notebook's fixed harness scraped weight memory with
`re.search(r"weights (?:took|take) ([\d.]+) GiB", log)`. Current vLLM logs `Model loading took 14.29 GiB`,
so the regex missed, `weights_gib` came back `NaN`, and the comparison table printed `nan` for the one
metric the first writeup question was about. The harness was marked do-not-edit.

**Technique.** Leave the harness alone and go to the artifact it was reading. The real numbers were sitting
in the saved server logs the whole time, so I sourced them from there, put them in the writeup, and named
the log line they came from. Then I shipped the logs *with* the submission so a reviewer could check the
claim against the same file.

**When to use.** Graded, audited, or otherwise frozen harnesses. More generally, any time a derived metric
is wrong but the raw evidence underneath it is intact: fix the reading, not the instrument.

**Code sketch.**
```python
# harness said nan; the log did not
w = {t: float(re.search(r"Model loading took ([\d.]+) GiB",
                        open(f"results/q2/vllm_{t}.log").read()).group(1))
     for t in ("bf16", "fp8")}          # {'bf16': 14.29, 'fp8': 8.2} -> 1.74x
```

**Pitfall.** The tempting move is a one-character fix to the regex, which silently invalidates the run under
the rules and is invisible in the diff a grader skims. Disclosure beats repair here: the `nan` stayed
visible, the writeup carried the true figures with their provenance, and the grader credited the analysis
as "grounded in actual vLLM log data".

**Source.** Quantization/serving homework, BF16 vs FP8. Predicted before the paid run, from reading the
harness regex against the log format the installed version actually emits.

---

## Assert every claim in the writeup against the artifacts with a script, not a read-through

**Problem.** A consistency pass done by eye is exactly as reliable as your attention at midnight, and my
writeup carried 34 numbers: measured medians, ratios I had rounded, tail percentiles pulled from raw
reports, and a parameter-accounting argument I had derived by hand.

**Technique.** Write the QA as executable assertions. Every figure that appears in the prose gets read back
out of the artifact it claims to come from, every ratio gets recomputed from source rather than trusted, and
every derived claim gets rebuilt from first principles. Extends "every number must trace to an output cell"
by making the trace mechanical.

**When to use.** Any deliverable where prose asserts numbers, especially once the compute that produced them
is gone. Run it as the last step, after the environment that made the numbers is already torn down.

**Code sketch.**
```python
near(1.74, w["bf16"] / w["fp8"], "memory ratio")           # recompute, don't retype
near(31.6, (1 - r_itl) * 100, "'32% faster decode'")       # check the rounding you wrote
# and re-derive the argument itself
pred = quant_params / GiB + emb_params * 2 / GiB           # 8.16 predicted
assert abs(pred - w["fp8"]) / w["fp8"] < 0.01              # 8.20 measured
```

**Pitfall.** Set the tolerance to the precision you actually wrote, or the script cries wolf. Two of my
"failures" were the checker's fault: a `…` that lived inside the template's own instruction comment, and
`4.1255` flagged against the `4.13` I had correctly rounded to. Investigate every flag rather than trusting
either the script or yourself, and record which ones were false.

**Source.** Quantization/serving homework. Also caught nothing real, which is the point: it converted "I
think the writeup is right" into 34 checks and a leak scan.

---

## Candor and self-documenting artifacts are rewarded, not penalized

**Problem.** Under review the instinct is to hide the wrong turns and ship only the clean result. It
feels safer. It scores worse, and it builds less trust.

**Technique.** Write the gotcha into the artifact and the misread into the report. On a graded MLOps
submission (97/100) the reviewer gave explicit credit for two things that feel like admissions of
weakness: (a) Grafana panel descriptions that warn the next reader about the traps ("these dashed lines
are thresholds, not data"; "tall P99 tails here are histogram artifacts"), and (b) a report that
candidly narrated two dashboard misreads I made and then corrected. Both were cited as adding "real
pedagogical value", and the honest "SLO missed, here is exactly why" outscored any green check.

**When to use.** Any reviewed or shared deliverable: a report, a dashboard, a PR, or a product's own
output to a user. Especially when a trap you hit will trip the next reader too.

**Pitfall.** Candor is not self-flagellation. Narrate the wrong turn, the correction, and what it taught,
in a line or two. A misread you found and fixed is a strength; one you hid is a landmine. And fix the
artifact, not just your notes, so the annotation travels with it.

**Source.** Text-to-SQL vLLM SLO — graded 97/100; Phase 2 and Phase 7 credit explicitly named the panel
descriptions and the candidly-described misreads.
