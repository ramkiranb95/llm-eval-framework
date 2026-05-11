# PR Review Checklist — LLM Evaluation Framework

Checklist for reviewing any pull request against this codebase.
Applies to all contributors and AI-assisted changes.
Sourced from: `codingStandards.md`, `conventions.md`, `testingPhilosophy.md`, and project decisions made during Tier 1 build.

---

## How to Use This

Work through each section top to bottom. Mark each item:
- `[x]` — passes
- `[ ]` — fails or not checked
- `[~]` — partially passes, comment added

A PR is ready to merge only when all critical items pass.
Non-critical items can be deferred with a documented reason.

---

## 1. Code Quality & Standards

### Naming — PEP 8 + Google Style Guide
- [ ] Python files: `snake_case.py`
- [ ] Functions: `snake_case()` — no leading underscores on non-private methods
- [ ] Private helpers: `_snake_case()` — single underscore only
- [ ] Classes: `PascalCase`
- [ ] Constants: `UPPER_SNAKE_CASE`
- [ ] Variables: `snake_case`
- [ ] Markdown docs: `camelCase.md`
- [ ] No cryptic abbreviations — names are self-explanatory

### Imports — PEP 8 § Imports
- [ ] One import per line — no `import os, sys`
- [ ] Order: standard library → third-party → internal, one blank line between groups
- [ ] No `from x import *`
- [ ] Lazy imports inside functions only when justified (e.g. optional heavy deps)

### Type Hints — PEP 484
- [ ] All public function signatures have type hints
- [ ] `Optional[X]` used wherever `None` is a valid return
- [ ] Return types declared

### Docstrings — PEP 257
- [ ] Every public function has a docstring
- [ ] Module docstring present at top of every file
- [ ] Docstring format: one-line summary → Args → Returns → Raises
- [ ] Imperative mood: "Return", "Load", "Evaluate" — not "Returns", "Loads"
- [ ] Private helpers have a minimum one-liner

### Error Handling — Google Style Guide § 2.4
- [ ] No bare `except:` — specific exceptions only
- [ ] Broad `except Exception` only when error is surfaced in return value
- [ ] No silent discards — errors are named and returned, not swallowed
- [ ] `ValueError` used for bad input at system boundaries
- [ ] `NotImplementedError` used to guard incomplete code paths

---

## 2. Module Structure

- [ ] Module docstring at top — includes Usage and Standalone test sections
- [ ] Section headers use `# ──` separator style
- [ ] Layout order: docstring → imports → constants → private helpers → public functions → `__main__`
- [ ] `__main__` block present and runnable independently
- [ ] `__init__.py` files remain empty — no re-exports added

---

## 3. Architecture & Design

### Config vs Code
- [ ] No hardcoded literals in source (timeouts, thresholds, delays, temperatures, lengths)
- [ ] All tunable values live in `config.yaml`
- [ ] Nothing reads `config.yaml` directly — all access through `config_loader.py` getters
- [ ] New config values have a getter function in `config_loader.py`

### Public API Contract
- [ ] New evaluator modules expose exactly `evaluate(pipeline_output: dict, config: dict) -> dict`
- [ ] No new public functions added without updating `conventions.md`
- [ ] Existing public API signatures unchanged unless explicitly discussed

### Directory Structure
- [ ] No code inside `config/`, `data/`, `docs/`, `reports/`
- [ ] No data files or configs inside `src/`
- [ ] No nesting beyond one level inside `src/`
- [ ] New files placed in the correct directory — no convenience dumping at root

### Score Conventions
- [ ] New metrics declare polarity in `threshold_checker.INVERTED_METRICS` (if lower = better)
- [ ] New metrics added to both `combined_evaluator.py` AND `config.yaml` — never one without the other
- [ ] Threshold range validated: `0.0 – 1.0` (enforced by `validate_config_thresholds()`)

---

## 4. File Paths & Concurrency

- [ ] File paths use `pathlib.Path` — not `os.path`
- [ ] Blocking calls with timeouts use `ThreadPoolExecutor` — not `threading.Thread`
- [ ] No `sleep()` loops for polling — use proper timeout patterns

---

## 5. Testing

- [ ] Test file named `test_<module>.py`
- [ ] Test functions named `test_<what_it_tests>`
- [ ] Shared fixtures in `conftest.py` — not duplicated across test files
- [ ] Data-driven tests use `@pytest.mark.parametrize`
- [ ] No mocking of data files or database — hit real JSON files
- [ ] `playground.py --all` runs cleanly after the change (primary validation)
- [ ] `pytest tests/` passes after the change (secondary validation)
- [ ] New evaluator metrics covered in `test_eval_cases.py`

---

## 6. LLM Evaluation Specific

### Metric Additions
- [ ] Metric definition sourced from RAGAs or DeepEval official docs — not invented
- [ ] Threshold calibrated for combined mode (single-pass scoring characteristics)
- [ ] `critical: true/false` set deliberately — critical metrics block the release gate
- [ ] Description in `config.yaml` matches the actual metric intent

### Evaluator Changes
- [ ] `combined_evaluator.py` prompt updated if metric definition changed
- [ ] `FRAMEWORK_MAP` updated if new metric added
- [ ] `LLM_METRICS` list updated if new metric added
- [ ] `INVERTED_METRICS` updated if new inverted metric added
- [ ] Disagreement detection logic still valid after metric changes

### Pipeline Changes
- [ ] Gate 1 (language check) still fires correctly for non-English inputs
- [ ] Gate 2 (empty context) still fires correctly for zero-context cases
- [ ] `generate_response()` `verbose` parameter respected — no print() outside the guard
- [ ] `inter_case_delay_seconds` appropriate for the active SUT provider's rate limits

---

## 7. Security & Secrets

- [ ] No `.env` file read or logged anywhere in changed code
- [ ] No API keys hardcoded — all keys referenced by name only (e.g. `GEMINI_API_KEY`)
- [ ] `.env` present in `.gitignore` — not staged
- [ ] No credentials, tokens, or personal data in any committed file

---

## 8. Version Control

- [ ] Commit message describes *why*, not *what the code does*
- [ ] No `--no-verify` flag used
- [ ] No amending of published commits
- [ ] Co-authored by Claude noted if AI-assisted

---

## 9. Documentation

- [ ] `learningLog.md` updated if a new issue, fix, or concept was introduced
- [ ] `codingStandards.md` updated if a new library or pattern was introduced
- [ ] `conventions.md` updated if a new public API or directory rule was introduced
- [ ] `config.yaml` comments updated if behaviour changed
- [ ] Stale comments removed — no "not yet implemented" left for implemented features

---

## 10. Impact Analysis (complete before approving)

Answer these before marking the PR ready:

| Question | Answer |
|---|---|
| Which files does this change depend on? | |
| Which files depend on the changed files? | |
| Does this change the public API of any module? | |
| Does this add a new config value? Is there a getter? | |
| Does this add a new metric? Is it in both evaluator and config? | |
| Could this break a running playground or pytest run? | |
| Does this touch rate-limit sensitive code paths? | |
| Does this touch the release gate logic? | |

---

## 11. Pre-Review Checklist — Before Raising the PR

Sourced from live project standards. Complete before requesting review.

- [ ] Read the task/ticket and acceptance criteria before writing any code
- [ ] Read existing files in the affected area before adding new code
- [ ] Searched for an existing function before writing a new one
- [ ] Confirmed no existing module can be extended instead of duplicated
- [ ] All open questions answered before coding — no assumptions made
- [ ] `playground.py --case TC001` (or relevant TC) passes locally
- [ ] `pytest tests/` passes locally
- [ ] No `print()` statements outside `if verbose:` guard
- [ ] No debug-only code left in committed files
- [ ] Stale comments removed — no "TODO", "not yet implemented", or dead blocks

---

## 12. Code Review Mindset — Universal Principles

These apply regardless of stack. Sourced from live project `CLAUDE.md` and test architecture experience.

### Before Writing Any Fix
- Read the complete error log before touching code
- Identify the exact failing line, what was received vs expected, and why — only then write the fix
- One fix per identified root cause — not a sequence of attempts

### Suggesting Alternatives
- When reviewing, actively check if a better approach exists before approving the obvious one
- When suggesting an alternative, always state: what the current approach does, why the alternative is better, and what risk the current approach carries if left unchanged

### Reuse Before Adding
- Search existing modules before writing a new function
- If a function exists in another module, extract it to a shared utility — do not duplicate
- If used in 3+ places, it belongs in a shared helper

### Root Cause Over Workarounds
- Never mask a failure with a retry or a skip without a documented reason
- When a test fails, fix the root cause — do not work around it
- If a fix is non-trivial, mark it explicitly and raise a follow-up item

### Minimum Blast Radius
- Scope changes to the smallest possible surface
- If a change can live in one module, keep it there — do not touch shared code unnecessarily
- Before modifying any shared module: list all files that import it, confirm blast radius with the reviewer

### One Purpose Per Unit
- One function does one thing
- One test asserts one behaviour with one set of inputs
- A setup step does not carry assertions from a verification step — keep them separate

### Verify Behaviour Before Asserting
- Do not write assertions based on what you expect the system to do
- Run the code, observe the actual output, then write the assertion
- This applies to both test assertions and score threshold assumptions

---

## Critical Items — PR Cannot Merge Without These

1. No hardcoded secrets or API keys
2. No bare `except:` swallowing errors silently
3. New metric added to both `combined_evaluator.py` AND `config.yaml`
4. `playground.py --all` runs cleanly
5. Commit message describes the why
6. No stale "not yet implemented" comments for implemented features
7. No open questions or assumptions — all context confirmed before the change was made
8. Root cause fixed — no masked failures or unexplained skips

---

## Non-Critical — Defer with Comment

- Type hints on private helpers (nice to have, not enforced)
- `learningLog.md` update (can follow in a separate commit)
- Full docstring on new private helpers (one-liner minimum is enough)
