# RAG — retrieval, reranking, and faithfulness

*Source: a RAG pipeline over FinanceBench 10-K / 10-Q filings. Llama-3.3-70B generation,
DeepSeek-V3.2 as judge, `bge-small-en-v1.5` embeddings, FAISS, all on Nebius. Baseline
(chunk 1000 / overlap 150, retriever k=4): correctness 0.29, faithfulness 0.41,
page_hit@5 0.38. The numbers below are real, measured in the notebook.*

---

## Measure retrieval separately from generation

**Problem.** Baseline correctness was 0.29. The instinct is to blame the LLM and swap models —
usually wrong.

**Technique.** Add a retrieval-only metric: `page_hit@k` — does any retrieved chunk land on the
gold evidence page? It isolates retrieval quality from generation quality.

**When to use.** Before tuning prompts or swapping models. If page_hit is low, the LLM literally
cannot answer; fix retrieval first.

**Finding.** page_hit@5 was only 0.38 — the gold page was missing on 62% of questions.
Generation is capped by what retrieval surfaces.

**Pitfall.** Dense retrieval on boilerplate-heavy corpora (10-K "non-GAAP" language) returns the
right *document* but the wrong *page*, and even pulls chunks from other companies' filings via
shared boilerplate.

**Source.** RAG / FinanceBench — `assignment2_rag.ipynb`.

---

## Raise k before reaching for anything fancier

**Problem.** Is the retriever returning too few chunks, or would more just dilute the context?

**Technique.** Re-run the eval with `k=8` instead of `k=4`. The cheapest possible change.

**Finding (real).** correctness 0.29 → 0.34 (+5pp), faithfulness 0.41 → 0.48 (+7pp),
page_hit@5 0.38 → 0.43. page_hit@1 and @3 stayed flat — proof the gains came from the newly
added chunks 5–8, not noise. The feared "dilution" never appeared.

**When to use.** When page_hit climbs with k (the right chunk is in the pool but ranked low).
Confirm @1/@3 stay flat — that tells you it's a ranking-depth win, not luck.

**Pitfall.** This only helps if the right page is already in the candidate pool. If page_hit@5 is
also flat, you have a recall problem and more chunks won't help.

**Source.** RAG / FinanceBench — `k=8` experiment.

---

## A domain-mismatched reranker makes things worse

**Problem.** "Add a cross-encoder reranker to promote the best chunks" sounds like free upside.

**Technique tested.** FAISS top-20 → `bge-reranker-base` cross-encoder → top-4.

**Finding (real, and surprising).** Every metric fell: correctness 0.29 → 0.24, faithfulness
0.41 → 0.28 (−13pp), page_hit@5 0.38 → 0.27 (−11pp). The dropping page_hit is the smoking gun —
a good reranker *raises* it; this one demoted the right chunks FAISS already had on top.

**When to use.** Only if the reranker is tuned for (or proven on) your domain. For keyword-dense
domains like finance, BM25 or a finance-tuned cross-encoder likely beats a generic neural one.

**Pitfall.** A reranker is not free upside — a domain-mismatched one actively destroys good dense
ranking. Always check page_hit before/after.

**Source.** RAG / FinanceBench — `reranker` experiment.

---

## A stricter prompt trades correctness for faithfulness

**Problem.** Can a tighter prompt make answers more grounded?

**Technique.** A strict system prompt: refuse (with an exact sentence) if the context doesn't
*directly* contain the answer; no inference or estimation; cite `[doc, page]` per claim; one
paragraph. Retrieval left unchanged.

**Finding (real).** faithfulness 0.41 → 0.47 (+6pp) but correctness 0.29 → 0.24 (−5pp). page_hit
stayed exactly at baseline — a clean confirmation that a prompt-only change does not move
retrieval metrics.

**When to use.** Strict prompts win only when a wrong answer costs more than a refusal (regulated,
high-stakes). On a correctness-graded benchmark the trade was net-negative.

**Pitfall.** Faithfulness and helpfulness are coupled, not independent dials — "refuse if context
is weak" also refuses correct answers that needed a small inferential bridge.

**Source.** RAG / FinanceBench — `strict_prompt` experiment.

---

## Grounding-only RAG is more honest, less informative — and confident-wrong is the real danger

**Problem.** Does RAG always beat a naive (no-retrieval) LLM? No.

**Finding.** RAG won on questions whose answer is literally in the corpus (e.g. a cash-drop
computed from two cited filings). But it *lost* on memorized-concept questions — a bank
gross-margin question that the naive model answered correctly from world knowledge, RAG refused
because the concept isn't in the chunks. Net: "more honest at the cost of being less informative."

**The dangerous case.** RAG once produced a confident citation to a real-but-wrong program. A
confidently-wrong cited answer is worse than a refusal — it looks authoritative.

**When to use.** Expect RAG to convert a naive model's confident bluffs into honest refusals on
out-of-corpus questions: a faithfulness win, a correctness/helpfulness loss. Budget for it, and
note that RAG does not fix the wrong-citation failure mode.

**Source.** RAG / FinanceBench — naive-vs-RAG comparison.

---

## Cache every LLM call to JSONL (resume/skip) for reproducible experiments

**Problem.** Each experiment re-runs ~100 generations + judge + faithfulness calls. Re-running
cells re-pays for everything and adds nondeterminism.

**Technique.** Append every result to a JSONL keyed by example id; on start, load the done ids and
skip them. One cache file per experiment (`cache_exp_k8`, `cache_exp_reranker`, …), each with a
paired `_faith` cache. A single `run_full_eval(label, retrieve_fn)` closure swaps only the
retrieval function. Generation async with a concurrency semaphore; `temperature=0`.

**When to use.** Any multi-experiment eval loop.

**Pitfalls (all learned here).**
- Faithfulness needs its **own** cache — the metric library makes its own (paid) LLM calls.
- Ragas refuses a sync `.score()` inside a notebook event loop; use `.ascore()`.
- Verify model IDs against the live catalog before each run — the specified judge snapshot was
  retired by the provider and had to be swapped.
- If you score faithfulness on a subsample (here, the first 20 rows) but correctness on all ~100,
  the faithfulness deltas are noisier — don't over-read small swings.

**Source.** RAG / FinanceBench — the `cache_*.jsonl` family and eval harness.

---

*Honest scope: chunk size (1000 / 150) was the assignment's fixed default, never swept — there are
no chunking before/after numbers. Hybrid BM25+dense, query rewriting, and per-document routing
were proposed but not implemented.*
