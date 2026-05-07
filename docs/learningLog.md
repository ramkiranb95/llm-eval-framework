# Learning Log — LLM Eval Framework

Lessons from building, testing, and debugging this project.
Added as they happen. Never delete — only append.

---

## L001 — Test parsers against non-default expected values

**Date:** 2026-04-27

**What happened:**
A regex parser had a bug — it required closing tags the LLM wasn't writing.
The bug was invisible because smoke testing only used a case whose expected
output matched the fallback default. `--all` revealed the failure.

**Lesson:** Always include at least one test case where the expected output
is NOT the fallback/default. If every passing case would pass even without
parsing, the test proves nothing.

---

## L002 — Async HTTP clients must be closed inside the event loop

**Date:** 2026-04-27

**What happened:**
An `AsyncOpenAI` client was created outside the async function, then
`asyncio.run()` closed the event loop. The client's connection pool tried
to clean up after the loop was gone → `RuntimeError: Event loop is closed`
on every connection. Scores were still correct but the noise masked real errors.

**Lesson:** Async clients must be created AND closed inside the same
`async def` that `asyncio.run()` drives. Always use a `try/finally` to close:
```python
async def _run():
    client = AsyncClient(...)
    try:
        # do work
    finally:
        await client.close()
asyncio.run(_run())
```

---

## L003 — Report saving must be crash-safe

**Date:** 2026-04-27

**What happened:**
`--save-report` did not create a new file because a RuntimeError mid-run
prevented the save logic from executing. Only the old report remained.

**Lesson:** Report saving must run inside `try/finally` so it executes even
when individual evaluators throw. After any `--save-report` run, verify a
new timestamped file exists in `reports/`.

---

## L004 — Substring matching cannot measure semantic coverage

**Date:** 2026-04-27

**What happened:**
`key_facts_coverage` used exact substring matching. The LLM always
paraphrased key facts — never quoted them verbatim — so every case scored 0.000.
Initially looked like a bug. It is a design limitation.

**Lesson:** Exact substring matching breaks on any real LLM output.
Semantic coverage needs embedding similarity or an LLM judge.
Until upgraded, keep this metric non-critical and treat 0.000 as expected.

---

## L005 — Always print raw LLM output before wiring a parser

**Date:** 2026-04-27

**What happened:**
A prompt format was designed with opening and closing delimiter tags.
The LLM followed the opening tags but consistently skipped the closing tags.
The parser was wired before the raw output was inspected.

**Lesson:** After any prompt format change, print the raw LLM response first.
Confirm the model actually follows the format before building parsing logic
around it. Never assume instruction-following without verifying it.

---

## L007 — LLMs drop structured output sections under prompt complexity

**Date:** 2026-04-29

**What happened:**
A prompt asked the model to produce free-text first, then a structured JSON block.
On simple cases it worked. On complex cases (longer replies, high-stakes content)
the model generated the free-text section and silently omitted the structured section.
The parser fell back to a default value, masking the real model behavior.

Two compounding issues in the parser:
1. Non-greedy regex `\{.*?\}` stops at the first `}` — breaks multi-line JSON silently.
2. Markdown code fences (` ```json...``` `) around JSON were not stripped before parsing.

**Fix:**
- Put format instructions and the output template immediately before where the model
  writes — not buried in the middle of a long prompt. Use `<placeholder>` markers
  in the template instead of inline instructions.
- Make the structured section mandatory with an explicit instruction ("do not skip").
- Parser: use greedy `.*` in JSON regex, strip code fences, log raw content on failure.

**Lesson:**
When a prompt has a long free-text section followed by a structured section, the
model prioritises the free-text and drops the structured section under token pressure.
Always inspect raw LLM output after any format change before assuming the parser works.
Test parsers against cases where the expected value differs from the fallback default.

---

## L008 — API quota `limit: 0` is an account-level block, not a rate limit

**Date:** 2026-04-29

**What happened:**
An API returned `429 RESOURCE_EXHAUSTED` with `limit: 0`. Adding a new key,
waiting past the rate-limit window, and switching model versions all produced
the same error. The issue was not temporary — `limit: 0` means no quota is
allocated to the account for that model tier, regardless of when or how many
calls were made.

**Fix:** Switched to a different provider entirely.

**Lesson:**
`limit: 0` is a provisioning problem, not a rate limiting problem. Retrying,
rotating keys, or waiting will not help. Validate the judge LLM with a one-liner
before integrating it into a pipeline — a 30-second check catches account-level
blocks before hours of debugging framework code.

---

## L009 — Silent code paths return wrong types and hide dead branches

**Date:** 2026-04-29

**What happened:**
An evaluation mode ("separate") was listed in config as a valid option but the
code path behind it only ran two of seven evaluators — the rest returned nothing.
The pipeline produced incomplete results without raising any error or warning.
The dead branch was only discovered during a structural audit, not from a test failure.

**Lesson:**
Every config-driven code path must either be fully implemented or explicitly
fail with a clear error (`NotImplementedError`, `ValueError`). Silent partial
execution is worse than a crash — it produces results that look valid but aren't.
If a feature is not yet built, guard it at the entry point so the user learns
immediately rather than debugging downstream data gaps.

---

## L010 — Standalone test scripts drift from the real config schema

**Date:** 2026-04-29

**What happened:**
Config keys were renamed in `config.yaml` (e.g. `primary_model` → `sut_model`,
`judge_mode` → `judge_provider`) but the `__main__` blocks in several modules
were still reading the old keys. The pipeline itself worked fine. The standalone
test scripts crashed. Since `__main__` blocks are only run manually they can go
undetected for a long time.

**Lesson:**
After any config schema change, grep for all field name references across
the codebase — not just pipeline code. Module-level `__main__` blocks, docstrings,
and documentation comments all reference config keys and must be kept in sync.
Treat standalone test scripts as a first-class consumer of the config schema.

---

## L011 — Config is the only source of truth — no hardcoded values in code

**Date:** 2026-05-02

**What happened:**
Phase A tech debt fixes initially had `timeout=30` and `temperature=0.0` hardcoded
directly in `ChatOpenAI(...)`. These belong in `config.yaml` so they can be changed
without touching code.

**Fix:**
Added `judge_timeout`, `judge_temperature`, `inter_case_delay_seconds` under the `llm`
block in `config.yaml`, and a new `pipeline` block for `min_email_body_length` and
`case_timeout_seconds`. `get_judge_config()` and the new `get_pipeline_config()` in
`config_loader.py` surface these to callers. Nothing is hardcoded in evaluators or
playground.

**Lesson:**
Any value a user might want to tune — timeouts, temperatures, delays, min lengths —
belongs in `config.yaml`. Code only reads it. If a value appears as a literal in
non-test code, it is a candidate for externalisation.

---

## L012 — Unused function parameters are a silent API smell

**Date:** 2026-05-02

**What happened:**
`_skipped_llm_scores(reason, eval_mode, config)` was written with two parameters
that were never accessed inside the function. The IDE flagged them immediately.
These came from over-engineering the signature before thinking about what the
function actually needed.

**Lesson:**
Write the function body first, then derive the signature from what is actually used.
Don't add parameters speculatively. IDE hints for unused parameters are free code reviews
— address them immediately before they accumulate.

---

## L013 — Embedding model choice is constrained by index-query consistency

**Date:** 2026-05-04

**What happened:**
The original spec called for Google embedding-001 (Gemini) via LangChain for ChromaDB.
When building Phase C (real ChromaDB), the question arose whether to follow the spec
or use all-MiniLM-L6-v2 (sentence-transformers) which was already in use for
key_facts_coverage in the custom evaluator.

**Decision:** Use all-MiniLM-L6-v2 for ChromaDB embeddings.

**Why:**
The embedding model must be identical at index build time and query time — mixing
models produces vectors in incompatible spaces and breaks retrieval entirely.
Gemini embeddings would add an API dependency to the index build step, meaning
the knowledge base cannot be rebuilt without network access and quota.
Since Groq is the judge LLM (not Gemini), there is no end-to-end consistency
argument for using Gemini embeddings. The retrieval quality difference does not
justify the operational dependency.

**Lesson:** When choosing an embedding model, the primary constraint is
consistency — same model must be used everywhere the index is touched.
The secondary constraint is operational: avoid external API dependencies on
operations that need to be reproducible locally (index builds, CI runs).

---

## L015 — BERTScore has a high absolute floor — use relative ranking, not absolute gates for discrimination

**Date:** 2026-05-07

**What happened:**
Tests that checked `bert_score_f1 < 0.82` for wrong/irrelevant replies failed.
A weather forecast reply scored 0.86 against a loan eligibility reference.
Two domain-adjacent financial replies (gold loan vs EMI failure) scored 0.84 against each other.
BERTScore (roberta-large) assigns 0.82–0.87 to almost any pair of grammatical English sentences
because roberta-large embeddings are dense and English sentences share structural/functional tokens.

**Fix:**
For wrong-reply discrimination: compare correct vs wrong reply scores against the same reference
(relative ranking). The correct reply should always rank higher — that assertion holds.
For absolute gating: BERTScore 0.82 threshold only catches empty output, non-English text,
or deeply incoherent replies — not subtle topic drift.

**Lesson:**
BERTScore absolute thresholds gate catastrophic failures.
Relative ranking (correct > wrong for same reference) is the meaningful discrimination test.
ROUGE discriminates better on surface topic mismatch — use both together.

---

## L016 — ROUGE requires comparable-length reference and candidate

**Date:** 2026-05-07

**What happened:**
ROUGE-L scored 0.093 for a full paragraph reply against a 7-word expected_reply summary
("Confirm eligibility and list required documents."). ROUGE-L threshold of 0.30 failed.
This is correct ROUGE behaviour — not a threshold calibration error.
ROUGE measures word overlap as a fraction of the reference length. A short reference
with different vocabulary from a long candidate produces near-zero scores by design.

**Fix:**
ROUGE threshold tests must use matched full-length reference and candidate strings.
For production use: ground_truth.expected_reply must be a complete reference reply,
not a summary phrase. Short summaries are only valid for BERTScore (semantic), not ROUGE (lexical).

**Lesson:**
Always match reference and candidate length when setting ROUGE thresholds.
If your ground truth is a short summary, use BERTScore only.
ROUGE is meaningful when reference and candidate are both complete responses.

---

## L014 — Dead config keys are silent lies

**Date:** 2026-05-06

**What happened:**
`disagreement_threshold: 0.15` existed in config.yaml under the `llm` block but no code read it.
An interviewer opening config.yaml could ask "what does this do?" and the honest answer was "nothing."
Either implement it or delete it — dead config is worse than no config because it implies
capability that doesn't exist.

**Fix:**
Implemented `_flag_disagreements()` in combined_evaluator.py. Faithfulness and hallucination
measure opposite things — a reply faithful to context should have low hallucination. If both
are high (or both low), the judge contradicts itself. The check: `abs(faithfulness + hallucination - 1.0) > threshold`.
A `disagreement_warning` key is added to both score dicts when flagged. Score is not changed —
only transparency is added for debugging.

Also added `get_disagreement_threshold()` to config_loader.py so the key has a proper accessor.

**Lesson:** Every key in config.yaml must be read somewhere in code. If it isn't, either build
the feature or delete the key. Config is a contract — unused keys break the contract silently.

---

## L006 — A model cannot reliably judge its own output

**Date:** 2026-04-28

**What happened:**
The initial setup used Ollama (mistral) as both the SUT (reply generator) and
the judge (evaluator). Hallucination scores came back 0.000 on every case —
too clean to be real. The model was rating its own output as perfect.

**Lesson:** Self-judging measures self-consistency, not correctness.
The SUT and judge must always be different models — ideally different model
families (e.g. local Ollama vs external API). If the same model generates
and evaluates, inflated scores are guaranteed and the evaluation is meaningless.
Rule: `SUT ≠ Judge`.

---

---

## L017 — playground.py is the primary test runner, not pytest

**Date:** 2026-05-07

**What happened:**
pytest test files were added (test_eval_cases.py) to wrap the 16 email cases,
but this created confusion about what the "real" tests are.

**Lesson:**
`playground.py --all` is the primary end-to-end test runner for this project.
It runs all 16 BSFI CRM cases through the full pipeline and is the first thing
to get working correctly. pytest files (test_pipeline_gates.py,
test_custom_evaluator.py, test_eval_cases.py) are supplementary — good for CI/CD
and reporting but secondary to the playground working correctly.
Priority order: playground working → pytest unit/integration tests.

---

## L018 — Every metric in LLM_METRICS must have a threshold in config.yaml

**Date:** 2026-05-07

**What happened:**
8 new metrics were added to `combined_evaluator.py` (tone_professionalism,
toxicity, non_advice, topic_adherence, bias, pii_leakage, role_adherence,
answer_similarity) but were never added to `config/config.yaml` evaluation
section. The playground showed "no threshold configured" for all of them —
scores appeared but pass/fail was never evaluated.

**Lesson:**
Adding a metric requires changes in TWO places:
1. `src/evaluators/combined_evaluator.py` — add to `LLM_METRICS` list and prompt
2. `config/config.yaml` — add threshold, critical flag, enabled, description

If a metric has no threshold in config, it scores but never passes or fails.
It looks active but contributes nothing to the release gate. Always add both
together. Checklist: metric in prompt → metric in LLM_METRICS → metric in config.

---

## L019 — Free tier API rate limits require SUT and judge on different providers

**Date:** 2026-05-07

**What happened:**
Both SUT (llama-3.1-8b-instant) and judge (llama-3.3-70b-versatile) were on
the same Groq API key. Each case makes 2 calls — SUT then judge — back to back.
Groq free tier is 6,000 TPM. The judge eval prompt is ~3,700 tokens. After the
SUT call, the judge call immediately hit the TPM limit with 429 errors on almost
every case.

**Lesson:**
When SUT and judge share a provider, they compete for the same rate limit pool.
Best split for free tier:
- SUT: Groq (fast, low token per call, 30 RPM)
- Judge: Gemini (1M TPM free, handles large eval prompts easily)
This avoids TPM contention entirely. Also: Gemini free tier has daily RPD limits
per model — if one model is exhausted, switch to another (e.g. gemini-2.5-flash-lite
when gemini-2.0-flash-lite is exhausted).

---

## L020 — language_check must divide by alpha chars, not total chars

**Date:** 2026-05-07

**What happened:**
language_check used `ascii_alpha / total_chars` to detect English. BSFI replies
contain Rs., %, numbers, punctuation — all non-alpha characters. This dragged
the ratio below 0.85 on every English reply, scoring 0.0 (non-English) for all
16 cases including clearly English text.

**Lesson:**
Divide by total alphabetic characters, not total characters:
`ascii_alpha / total_alpha`
Punctuation, digits, currency symbols are not language indicators and must not
penalise the ratio. This applies to any text heuristic that measures character
class ratios in domain-specific content (financial, medical, legal).
