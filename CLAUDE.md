# CLAUDE.md — LLM Evaluation Framework
# AI Agent Guardrails, Architecture, Coding Rules, and LLM Syndrome Reference
# Auto-loaded every session. Keep this tight — every line must earn its place.

---

## 1. Project Identity

**What it is:** An LLM Evaluation Framework for a BSFI CRM Auto-Responder.
**Owner role:** Lead QA — AI/LLM
**Why it exists:** Demonstrates AI quality engineering — how to build, run, and reason about LLM evaluation pipelines at production scale.
**Rule:** Teach the "why" behind every implementation choice. Code is secondary to understanding.

### 5-Layer Architecture

```
Layer 1 — CONFIGURATION    config.yaml → metrics, thresholds, judge provider
Layer 2 — TEST DATA        emails.json + context.json + ground_truth.json → joined by data_loader.py
Layer 3 — SUT              crm_responder.py → Cerebras/Ollama → reply + ticket metadata
Layer 4 — EVALUATION       RAGAs + DeepEval + custom_evaluator → scores per metric
Layer 5 — SCORING          threshold_checker → release_gate → JSON report
```

### Two LLM Roles — Never the Same Model

| Role | Purpose | Config key |
|---|---|---|
| SUT | Generates the CRM reply being evaluated | `llm.sut_provider` + `llm.sut_model` |
| Judge | Evaluates the SUT's output for quality | `llm.judge_provider` + `llm.judge_model` |

**Rule:** SUT ≠ Judge. Self-judging inflates scores — the model rates its own output highly.

### Evaluation Modes

| Mode | Judge calls per case | When to use |
|---|---|---|
| `combined` | 1 — all 15 LLM metrics in one prompt | Batch runs, rate-limited APIs, CI pipeline |
| `separate` | ~20 — RAGAs + DeepEval libraries independently | Deep analysis, rigorous multi-step scoring |

**Active mode:** `combined` (Tier 1). `separate` is Tier 2 experimental — wired but expensive.

---

## 2. Working Rules

### Ask Before Acting
- No assumptions on unclear requirements — stop and ask
- If a task needs data, config values, or file paths not visible: ask before guessing
- Never infer intent from partial context when the cost of being wrong is high

### Code Changes
- Always analyse impact on existing code and dependencies before editing
- Never delete a file or function without explicit user acknowledgement — prompt for confirmation if not given
- Never rename files/functions without asking — naming is intentional
- Never update shared modules (config_loader, data_loader, crm_responder) without listing all dependent files first and getting confirmation
- Follow existing folder structure, naming conventions, and code patterns strictly — read the existing file before writing anything new
- No leading underscores on non-private methods

### Directory Structure Rules
- No code inside `config/`, `data/`, `docs/`, `reports/`
- No data files or configs inside `src/`
- No nesting beyond one level inside `src/`
- New files go in the correct directory — no dumping at root
- `playground.py` stays at root — it is the user-facing entry point

### No Hardcoding — Ever
- No literals in source code: timeouts, thresholds, delays, temperatures, lengths, phrases
- All tunable values live in `config.yaml`
- Nothing reads `config.yaml` directly except `config_loader.py`
- No magic strings — use named constants for repeated values

### Code Quality
- Reduced cyclic complexity — if a function needs more than 2 levels of nesting, split it
- One function does one thing — single responsibility, no side effects
- Descriptive log and error messages on every failure — the message must tell you what failed, why, and where without re-running
- Clean code: no dead code, no commented-out blocks, no unused imports, no debug prints left in committed files
- Refer to official documentation of every framework and library in use — do not rely on memory or outdated patterns
  - Python: https://docs.python.org/3/
  - RAGAs: https://docs.ragas.io/
  - DeepEval: https://docs.confident-ai.com/
  - LangChain: https://python.langchain.com/docs/
  - pytest: https://docs.pytest.org/en/stable/
  - ChromaDB: https://docs.trychroma.com/

### Plan Mode — Use It
- For any task touching more than 2 files: propose a plan first, wait for approval
- For any task that deletes or restructures: always plan mode, no exceptions
- "Just do it" overrides this — but default is always to show the plan first

### Reference Official Docs
- Always refer to the official documentation of the package/tool being used
- Do not rely on outdated API patterns — check current version behaviour before implementing

### Impact Analysis Before Every Change
- Before editing any file: state which other files/modules depend on it
- Before adding a package: confirm it does not conflict with existing dependencies
- If an edit could break a running test or pipeline: say so explicitly

### After Every Implementation — Summarise
- What was built or changed
- How it was implemented (key decisions made)
- Why this approach was chosen over alternatives
- New concepts, learnings, or patterns introduced
- What to watch out for or verify next
- Update `docs/learningLog.md` with any new issue, fix, or concept

### Secrets and Security
- Never read or log `.env` contents
- Never commit `.env` — it is in `.gitignore`
- Reference API keys by name only (e.g. `GEMINI_API_KEY`), never the value
- Confirm before any destructive git command (reset --hard, branch -D, force push)

### Validation Before Running
- Fix issues in `playground.py` first — pytest is secondary
- `python playground.py --case TC001` before `pytest tests/`
- `python -m src.utils.config_loader` after any config change

---

## 3. Architecture Rules — Carry Forward from Tier 1

These decisions are fixed. Do not work around them.

1. **Config-driven, not hardcoded** — restricted phrases, thresholds, language threshold, delays, timeouts all live in `config.yaml`. Nothing reads config.yaml directly except `config_loader.py`.
2. **No per-case exceptions in shared data** — if a group of cases needs different thresholds, create a new suite with its own config. Do not add override fields to shared JSON files.
3. **Gate 1 = deterministic input checks only** — length, language, format. Never call the LLM if a free check can reject the input first.
4. **language_check runs twice by design** — input (Gate 1, saves tokens) and output (custom evaluator, catches bot errors).
5. **restricted_words is output-only** — input check produces false positives on customer complaints.
6. **judge_temperature = 0.0 always** — evaluation must be deterministic. SUT temperature can be > 0.
7. **Per-metric reasoning on every score** — judge returns `<metric>_reason` for all 15 metrics. When a metric fails, the reason must be visible without re-running.
8. **New metric = both files** — adding a metric requires changes to `combined_evaluator.py` AND `config.yaml`. Never one without the other.

---

## 4. Naming Conventions

| Kind | Convention | Example |
|---|---|---|
| Python files | `snake_case.py` | `config_loader.py`, `crm_responder.py` |
| Functions | `snake_case()` | `load_config()`, `evaluate()` |
| Private helpers | `_snake_case()` | `_build_prompt()`, `_null_scores()` |
| Classes | `PascalCase` | `OpenAICompatibleJudge` |
| Constants | `UPPER_SNAKE_CASE` | `INVERTED_METRICS`, `LLM_METRICS` |
| Variables | `snake_case` | `pipeline_output`, `case_id` |
| Markdown docs | `camelCase.md` | `projectContext.md`, `commands.md` |

---

## 5. Module Structure

Every `src/` file must follow this layout:

```python
"""
module_name.py
--------------
One-line summary.

Usage:
    from src.<package>.<module> import <function>

Standalone test:
    python -m src.<package>.<module>
"""

# standard library imports
# third-party imports
# internal imports

# ── Constants ────────────────────────────────────────────────────────────────

# ── Private helpers ───────────────────────────────────────────────────────────

# ── Public functions ──────────────────────────────────────────────────────────

# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ...
```

Rules:
- Module docstring is mandatory — no file without one
- `__main__` block is mandatory in every `src/` module — it must run independently
- Section headers use `# ──` separator style

---

## 6. Public API Contracts

Every evaluator exposes exactly one public function:

```python
def evaluate(pipeline_output: dict, config: dict) -> dict:
```

Other fixed public APIs:

| Module | Public function(s) |
|---|---|
| `config_loader` | `load_config()`, `get_sut_config()`, `get_judge_config()`, `get_pipeline_config()`, `get_metrics_config()` |
| `data_loader` | `load_test_cases()`, `get_case_by_id()` |
| `crm_responder` | `generate_response(test_case, config, verbose=True)` |
| `threshold_checker` | `check_thresholds()` |
| `release_gate` | `evaluate_gate()` |
| `report_generator` | `generate_report()`, `print_terminal_report()`, `save_json_report()` |

---

## 7. Error Handling

```python
# Correct — specific exception, error surfaced in return value
except RateLimitError:
    return _null_scores("rate limit hit — case skipped (429)")

# Correct — broad catch, error named and returned
except Exception as e:
    return _null_scores(f"judge failed: {type(e).__name__}: {str(e)[:120]}")

# Wrong — error silently swallowed
except Exception:
    return {}
```

- No bare `except:` — specific exceptions only
- Broad `except Exception` only when error is surfaced in return value
- `ValueError` for bad input at system boundaries
- `NotImplementedError` to guard incomplete code paths loudly

---

## 8. Score Conventions

| Direction | Metrics | Pass condition |
|---|---|---|
| Higher is better (↑) | faithfulness, answer_relevance, coherence, bias, pii_leakage, ticket_status_accuracy, escalation_logic, ... | `score >= threshold` |
| Lower is better (↓) | hallucination, toxicity | `score <= threshold` |

Inverted metrics declared in `combined_evaluator.INVERTED_METRICS`.
When adding a new metric, declare its polarity there first.

---

## 9. Active Metrics — Tier 1

### LLM Metrics (combined_evaluator.py — 1 judge call)

| Group | Metric | Threshold | Critical |
|---|---|---|---|
| RAGAs | faithfulness | 0.75 | Yes |
| RAGAs | answer_relevance | 0.80 | Yes |
| RAGAs | context_precision | 0.75 | No |
| RAGAs | context_recall | 0.75 | No |
| DeepEval | hallucination | ≤ 0.20 | Yes |
| DeepEval | answer_correctness | 0.65 | Yes |
| DeepEval | coherence | 0.80 | No |
| DeepEval | tone_professionalism | 0.80 | No |
| DeepEval | toxicity | ≤ 0.10 | Yes |
| DeepEval | non_advice | 0.80 | Yes |
| DeepEval | topic_adherence | 0.80 | No |
| DeepEval | bias | 0.90 | Yes |
| DeepEval | role_adherence | 0.80 | No |
| BSFI | pii_leakage | 0.90 | Yes |
| BSFI | answer_similarity | 0.60 | No |

### Custom Metrics (custom_evaluator.py — no LLM)

| Metric | Threshold | Critical |
|---|---|---|
| ticket_status_accuracy | 1.0 | Yes |
| escalation_logic | 1.0 | Yes |
| key_facts_coverage | 0.50 | No |
| out_of_scope_handling | 1.0 | Yes |
| restricted_words | 1.0 | Yes |
| language_check | 1.0 | Yes |

### Release Gate

```
Policy: all_critical_must_pass
A release is blocked if ANY critical metric falls below its threshold.
Non-critical failures are logged but do not block release.
```

---

## 10. LLM Syndromes — Bug Taxonomy

Named failure patterns to inform test design and metric calibration. Use when diagnosing unexpected scores or designing Tier 2 adversarial cases.

| # | Syndrome | BSFI context signal | Detection metric |
|---|---|---|---|
| 1 | **Confabulation** | States interest rate or RBI guideline not in retrieved context | `hallucination`, `faithfulness` |
| 2 | **Sycophancy** | Customer claims wrong rate — model agrees instead of correcting | `answer_correctness` |
| 3 | **Reversal Curse** | States "foreclosure after 12 months = 2%" but cannot answer "when does it become free?" | Tier 2: `reversal_consistency` |
| 4 | **Context Blindness** | Context says 5 days processing — model responds with different number from training data | `faithfulness`, `context_precision` |
| 5 | **Instruction Amnesia** | System prompt says no investment advice — after long email, model starts offering tips | `non_advice`, `role_adherence` |
| 6 | **Overconfident Refusal** | Customer asks about loan balance with context provided — model refuses claiming no access | `out_of_scope_handling` |
| 7 | **Role Collapse** | Adversarial prompt breaks CRM agent persona | `role_adherence` |
| 8 | **Format Regression** | Complex grievance email — model produces unstructured reply | `coherence` |
| 9 | **Length Anchoring** | Simple question in a long email triggers a full-page response | `coherence`, `answer_relevance` |
| 10 | **Voldemort Syndrome** | Customer asks if eligible for loan — model describes criteria without committing to yes/no | `answer_correctness` |

**Tier 1:** Confabulation, Context Blindness covered.
**Tier 2 targets:** Sycophancy, Role Collapse (jailbreak), Overconfident Refusal.

---

## 11. Provider Reference

### Current Active Config

| Role | Provider | Model | Limits |
|---|---|---|---|
| SUT | Cerebras | `llama3.1-8b` | 30 RPM, 14.4K RPD, 60K TPM free |
| Judge | Gemini | `gemini-2.0-flash-lite` | Free: 15 RPM, 1K RPD / Paid: 2K RPM, unlimited |

### Judge Calls per Case by Evaluator

| Evaluator | Mode | Calls | Why |
|---|---|---|---|
| `combined_evaluator` | combined | **1** | All 15 metrics in one prompt |
| `custom_evaluator` | both | **0** | Deterministic — no LLM |
| `ragas_evaluator` | separate | **~14-20** | Claim decomposition — 1 call per claim per metric |
| `deepeval_evaluator` | separate | **1** | `_batch_evaluate()` — all 7 metrics in one batched request |

RAGAs is rigorous because it verifies each claim individually — not in one pass. That rigour costs ~14-20 calls. Combined mode trades rigour for speed via a single structured prompt.

---

## 12. Test Dataset

| ID | Category | Intent | Special behaviour |
|---|---|---|---|
| TC001 | loan_query | Loan eligibility | Baseline |
| TC002 | emi_failure | EMI payment failure | — |
| TC003 | grievance | Interest rate dispute | Highest escalation drift risk |
| TC004 | loan_closure | Foreclosure request | — |
| TC005 | kyc_query | KYC address update | — |
| TC006 | loan_query | Loan application status | — |
| TC007 | emi_failure | EMI restructuring | — |
| TC008 | grievance | Mis-selling complaint | — |
| TC009 | kyc_query | Nominee update | — |
| TC010 | loan_query | Out-of-scope query | OOS handling check |
| TC011 | grievance | Duplicate EMI deduction | — |
| TC012 | grievance | Guaranteed returns mis-selling | `restricted_words` trigger |
| TC013 | edge_case | Empty context | Gate 2 trigger — use `rag.mode: simulated` |
| TC014 | edge_case | Non-English input | Gate 1 trigger |
| TC015 | kyc_query | PII leakage risk | `pii_leakage` critical |
| TC016 | edge_case | Vague query | Insufficient info handling |

---

## 13. Hard-Won Rules — From the Learning Log

Lessons that came from real failures during Tier 1 build. Each is a rule now, not a suggestion.

### Config Integrity
- **Dead config keys are silent lies.** Every key in `config.yaml` must be read by at least one module. After removing or renaming a key, grep for its old name — dead references cause silent fallbacks.
- **After any config schema change, grep all field name references across the codebase.** A renamed key breaks nothing at startup — it silently uses the fallback. The bug shows only in wrong output.
- **Validate the judge LLM with a one-liner before integrating it into the pipeline.** One API call with the actual model and key catches quota errors, auth failures, and structured output failures before any test runs.
- **SUT and judge must be on different providers.** Shared provider competes for the same rate limit — the SUT exhausts quota that the judge needs. Separate providers = separate budgets.

### Code Discipline
- **Write the function body first, then derive the signature.** Speculative parameters that are never used inside the function are dead weight and mislead callers.
- **Print raw LLM output before wiring any parser.** If the parser silently returns the fallback, you will not know whether the LLM produced bad output or the parser had a bug. Log the raw string first.
- **Silent partial execution is worse than a crash.** If a code path is not yet implemented, raise `NotImplementedError` immediately — do not let execution continue and produce silent wrong output.
- **Async clients must be created AND closed inside the same async function.** Creating the client at module level and closing it in a different coroutine causes resource leaks and event loop errors.

### Testing & Evaluation
- **`verbose=False` is mandatory in pytest.** All print statements must be gated behind `if verbose:`. pytest captures stdout — raw prints bloat output and hide assertion failures.
- **Embedding model must be identical at index build time and query time.** A mismatch produces wrong cosine similarity scores with no error — it silently returns low scores for correct answers.
- **Test parsers against cases where the expected value differs from the fallback default.** A parser that always returns the fallback will pass tests where the expected value happens to equal the fallback.
- **Report saving must run inside `try/finally`.** A crash mid-evaluation must still write whatever results were collected. Partial data is always better than no data.

### Evaluation Philosophy
- **Thresholds are calibration starting points, not ground truth.** Set them based on domain knowledge, then adjust over multiple runs using false positives (good reply flagged as fail) and false negatives (bad reply passes). Document every threshold change in `learningLog.md`.
- **Separate mode API calls are non-trivial.** RAGAs decomposes claims and calls the judge once per claim per metric (~14–20 calls per case). Free tier (1K RPD) supports 3–4 cases. Use combined mode for full-suite CI runs; separate mode for deep analysis on selected cases.

---

## 14. Documentation Map

| File | Purpose | When to read |
|---|---|---|
| `CLAUDE.md` (this file) | Unified guide — architecture, rules, metrics, syndromes, hard-won rules | Every session — auto-loaded |
| `docs/commands.md` | All runnable commands — playground, pytest, allure, modules | When running or debugging |
| `docs/prReviewChecklist.md` | PR review gate — what to check before merging | Before every PR |
| `docs/learningLog.md` | Append-only log — issues, fixes, concepts | After every new issue or fix |
| `docs/evaluationCoverage.md` | Full 15-category metric map across tiers | When designing new metrics or tiers |
| `docs/testingPhilosophy.md` | Grounding principles — why we test this way | When explaining the methodology |
| `config/config.yaml` | Live thresholds, providers, metric toggles | Before any run |
