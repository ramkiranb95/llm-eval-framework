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
