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
