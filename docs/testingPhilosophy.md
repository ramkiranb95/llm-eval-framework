# Testing Philosophy — LLM Evaluation Framework

## Grounding Principles

This framework is built on the premise that LLM testing is fundamentally
exploratory, evidence-based, and judgment-driven — not script-driven.

Metrics and thresholds are guardrails, not replacements for thinking.

---

## Thought Leaders and Their Influence

### James Bach & Michael Bolton — Rapid Software Testing (RST)

**Core idea:** Testing is a skilled human activity. The tester's job is to
find information that matters, not to execute scripts.

**Applied here:**
- LLM outputs are explored, not just asserted
- Failures are documented as evidence, not just pass/fail counts
- The framework generates data for human judgment, not verdicts

**Key RST principles used:**
- Test oracles must be defined explicitly (ground truth in each test case)
- Variation and exploration surface bugs that repetition misses
- Checklists are thinking tools, not compliance checklists

**Relevant work:** [Rapid Software Testing](https://www.satisfice.com/rapid-software-testing)

---

### Michael Bolton — LLM Syndrome Taxonomy

Bolton extended RST thinking to AI/LLM systems, naming failure patterns
as "LLM Syndromes" — observable, repeatable failure modes in language models.

These syndromes are documented in `llmSyndromes.md` and form the basis
for adversarial test cases in Tier 2.

**Key insight:** LLMs don't have bugs the way software does.
They have *tendencies* — statistical biases toward certain failure patterns.
Testing must account for this probabilistic nature.

---

### Rahul Parwal — AI in Testing vs Testing in AI

**Core distinction:**
- **AI in Testing** — using AI tools to assist testers (test generation, analysis)
- **Testing in AI** — evaluating AI systems as the product under test

This framework does both:
- Uses Claude API to generate synthetic test data (AI in Testing)
- Evaluates the CRM auto-responder LLM output (Testing in AI)

**Key principle:** Value is upstream — defining what "good" looks like
(the ground truth, thresholds, and evaluation criteria) is the hardest
and most important part. The code is secondary.

**Agentic QA formula:**
```
Agentic QA = Agentic Execution × Human Judgment
```
Automation scales execution. Judgment cannot be automated.

**Relevant work:** [Rahul Parwal on AI Testing](https://www.rahulparwal.com)

---

### Maaret Pyhäjärvi — Exploratory Testing and AI

**Core idea:** AI is an external imagination for exploratory testing.
It generates possibilities the tester might not have considered.

**Applied here:**
- Synthetic test cases (TC001–TC010) represent Claude's imagination
  of what a real BSFI CRM receives
- The tester's job is to evaluate whether these cases are realistic,
  complete, and representative — not to blindly trust generation

**Key principle:** Intent matters more than code.
A test without a clear intent is just noise.

**Insight on agentic testing:** Exploratory testing approaches work
well for agentic systems because agent behavior is non-deterministic
and context-dependent — the same prompt can produce different outcomes.

**Relevant work:** [Maaret Pyhäjärvi Blog](https://maaretp.com)

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
> — Dijkstra (still applies to LLMs)

---

## Calibration Approach

Thresholds in `config.yaml` are initial estimates.
As the framework runs against real outputs, adjust thresholds based on:

1. False positives — tests failing on acceptable outputs → raise threshold
2. False negatives — tests passing on bad outputs → lower threshold
3. Domain feedback — what matters to the business → reweight critical flags

This is an ongoing loop, not a one-time setup.
