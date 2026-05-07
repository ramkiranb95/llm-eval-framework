# Testing Philosophy — LLM Evaluation Framework

## Grounding Principles

This framework is built on the premise that LLM testing is fundamentally
exploratory, evidence-based, and judgment-driven — not script-driven.

Metrics and thresholds are guardrails, not replacements for thinking.

---

## Core Testing Principles

### Testing is a skilled human activity

The tester's job is to find information that matters, not to execute scripts.

Applied here:
- LLM outputs are explored, not just asserted
- Failures are documented as evidence, not just pass/fail counts
- The framework generates data for human judgment, not verdicts

Key principles used:
- Test oracles must be defined explicitly (ground truth in each test case)
- Variation and exploration surface bugs that repetition misses
- Checklists are thinking tools, not compliance checklists

---

### LLM failures are probabilistic tendencies, not deterministic bugs

LLMs don't have bugs the way software does. They have *tendencies* —
statistical biases toward certain failure patterns under certain conditions.
Testing must account for this probabilistic nature.

These failure patterns are documented as LLM Syndromes in `llmSyndromes.md`
and form the basis for adversarial test cases in Tier 2.

---

### AI in Testing vs Testing in AI

Two distinct disciplines — this framework does both:

- **AI in Testing** — using AI tools to assist testers (test generation, analysis)
- **Testing in AI** — evaluating AI systems as the product under test

This framework uses Claude API to generate synthetic test data (AI in Testing)
while evaluating the CRM auto-responder LLM output (Testing in AI).

Key principle: Value is upstream — defining what "good" looks like
(the ground truth, thresholds, and evaluation criteria) is the hardest
and most important part. The code is secondary.

Agentic QA formula:
```
Agentic QA = Agentic Execution × Human Judgment
```
Automation scales execution. Judgment cannot be automated.

---

### Exploratory testing applies to agentic systems

AI is an external imagination for exploratory testing.
It generates possibilities the tester might not have considered.

Applied here:
- Synthetic test cases (TC001–TC010) represent what a real BSFI CRM receives
- The tester's job is to evaluate whether these cases are realistic,
  complete, and representative — not to blindly trust generation

Intent matters more than code. A test without a clear intent is just noise.

Exploratory approaches work well for agentic systems because agent behavior
is non-deterministic and context-dependent — the same prompt can produce
different outcomes.

---

## How This Philosophy Shapes the Framework

| Principle | How It's Applied |
|---|---|
| Explicit oracles | Every test case has a defined `ground_truth` |
| Evidence over verdicts | Scores are stored, not just pass/fail |
| Human-readable config | `config.yaml` keeps thresholds visible and adjustable |
| Tier structure | Tier 1 proves the framework; Tier 2 explores edge cases |
| Release gate | A deliberate go/no-go gate — not just a test counter |
| Domain-specific checks | Custom evaluators capture what generic metrics miss |

---

## What This Framework Does Not Claim

- It does not claim to fully verify LLM correctness
- It does not replace human review of LLM outputs
- Passing all metrics does not mean the system is safe to deploy
- Thresholds are starting points — they require calibration over time

> "Testing shows the presence of defects, not their absence."

---

## Calibration Approach

Thresholds in `config.yaml` are initial estimates.
As the framework runs against real outputs, adjust thresholds based on:

1. False positives — tests failing on acceptable outputs → raise threshold
2. False negatives — tests passing on bad outputs → lower threshold
3. Domain feedback — what matters to the business → reweight critical flags

This is an ongoing loop, not a one-time setup.
