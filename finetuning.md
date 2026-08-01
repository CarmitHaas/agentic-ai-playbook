# Fine-tuning — QLoRA, small models, and judging the result

*Source: QLoRA fine-tuning of `SmolLM2-360M-Instruct` (causal) and `flan-t5-small` (seq2seq)
for level-adaptive QA (child/student/expert) on a dataset generated with Llama-3.1-8B, evaluated
with an LLM judge against a 20-question test set
([repo](https://github.com/CarmitHaas/qlora-level-adaptive-qa)). Graded 100/100 — because the
analysis disagreed with the judge, not despite it.*

---

## Keep the LoRA params in a dtype the optimizer can survive

**Problem.** QLoRA quantizes the base model to 4-bit, but the trainable adapter params and the
grad scaler still have opinions. On a T4 with fp16 autocast, fp16 LoRA params make the grad
scaler produce NaNs/overflow; T5-family models are famously unstable in fp16 end to end.

**Technique.** Base model in NF4 4-bit (+ double quant), LoRA adapter params cast to fp32 so the
fp16 grad scaler has headroom. For flan-t5, skip mixed precision entirely and train fp32 — the
model is small enough that stability is worth the memory.

**When to use.** Any QLoRA run on consumer/free GPUs. The quantization config and the training
dtype are two separate decisions; make both explicitly.

**Code sketch.**
```python
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)
model = prepare_model_for_kbit_training(model)   # upcasts norms etc.
for p in model.parameters():
    if p.requires_grad: p.data = p.data.float()  # fp32 LoRA params under fp16 scaler
```

**Pitfall.** The failure is silent-ish: loss goes NaN a few hundred steps in, and the first
suspect is always the learning rate, not the dtype of 0.4% of the parameters.

**Source.** QLoRA level-adaptive QA — Task 1.2.

---

## A LoRA recipe is per-architecture, not universal

**Problem.** The same "r=8, alpha=16, target q,v" recipe silently does different things on
different architectures — module names differ, and seq2seq models put half their capacity in the
decoder cross-attention.

**Technique.** Pick `target_modules` per family (causal: `q_proj,v_proj,...`; T5: `q,v` inside
both self- and cross-attention blocks), and expect different outcomes: the causal 360M learned the
level-conditioning; flan-t5-small never did — the level token entered the input and nothing
downstream could act on it.

**When to use.** Any time a fine-tune is compared "fairly" across architectures — the recipe being
nominally identical does not make the comparison controlled.

**Pitfall.** `peft` won't error on a target-module name that matches nothing in one of the two
models — it trains an adapter over a subset of what you intended and the run "works".

**Source.** QLoRA level-adaptive QA — Task 1.2 / 1.4.

---

## A judge comparing against a broken baseline measures "less broken", not "good"

**Problem.** The LLM judge said flan-t5 improved on 18/20 test questions — the highest score in
the eval. The fine-tuned model's actual answers included "location" for the capital of Kansas and
the literal string "(ii)" for three different questions.

**Technique.** Keep the judge, but make it produce a per-row table (question, base output, FT
output, expected, verdict, notes) and re-read the table yourself. The base flan-t5 was emitting
"neo-neo-neo"-grade garbage, so *any* coherent token counts as "better after FT" — the verdict is
relative to the baseline, and a broken baseline inflates everything. Honest re-count: 2–3/20.

**When to use.** Every before/after fine-tuning eval. "Better than base" is the wrong question
when base is unusable; also ask "acceptable in absolute terms?"

**Pitfall.** The judge's verdict distribution (18/20!) looks like your strongest result and goes
straight into the summary slide. The per-row table is the only thing that catches it — and
disagreeing with your own headline number in writing is what a grader (or a stakeholder) actually
rewards.

**Source.** QLoRA level-adaptive QA — Tasks 1.3 / 1.4.

---

## Small-model fine-tuning changes the register, not the knowledge

**Problem.** Expecting a 360M fine-tune to fix factual errors. After fine-tuning, SmolLM2 still
answered "the capital of Florida is Miami" and invented "Air Netherlands" as the Dutch flag
carrier.

**Technique.** Diff base vs FT outputs on the *same* questions. The Netherlands-airline answer was
byte-identical before and after fine-tuning — the adapter reshaped style (shorter child answers,
learned to stop instead of generating runaway `### Question:` continuations) while the facts
underneath never moved.

**When to use.** Setting expectations for any small-scale SFT: it is register/format/persona
adaptation. Facts come from the base model; fixing them needs retrieval, a bigger base, or
knowledge-targeted data — not 300 style examples.

**Pitfall.** Style improvement reads as overall improvement. Fluent-and-wrong is the worst
quadrant, and fine-tuning moved the model *into* it exactly as often as before.

**Source.** QLoRA level-adaptive QA — Task 1.4.

---

## Generate the dataset with a bigger model, then defend it like production input

**Problem.** Synthetic instruction data from an API model arrives with malformed JSON, refusals,
level bleed-through, and train/test leakage — and every flaw becomes a training signal.

**Technique.** Generation pipeline with: a model-fallback list (the primary ID can be retired or
throttled mid-run), retry with backoff, JSON schema validation on every sample, a balanced test
set constructed first (7/7/6 across the three levels), and an assert that no test question appears
in train.

**When to use.** Any dataset built by prompting a stronger model — which is most fine-tuning
datasets now.

**Pitfall.** The overlap assert feels paranoid until it fires. With both sets sampled from the
same filtered dolly pool, collisions are the default outcome, and a leaked test question turns the
eval into a memory test.

**Source.** QLoRA level-adaptive QA — Task 1.1.
