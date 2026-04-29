# Project Conventions

Conventions for the LLM Evaluation Framework. All contributors and AI tools
must follow these strictly. When in doubt, match the existing code — don't
introduce a new pattern without updating this document.

---

## File Naming

| Type | Convention | Examples |
|---|---|---|
| Python source files | `snake_case.py` | `config_loader.py`, `crm_responder.py` |
| Markdown docs | `camelCase.md` | `projectContext.md`, `conventions.md` |
| JSON data files | `snake_case.json` | `emails.json`, `ground_truth.json` |
| Config files | `snake_case.yaml` | `config.yaml` |
| Test files | `test_<module>.py` | `test_crm_responder.py` |

---

## Directory Structure

```
llm-eval-framework/
├── config/           # YAML configuration only — no code
├── data/             # Test data files only — no code, no generated output
├── docs/             # Markdown reference docs only
├── reports/          # Generated JSON reports (git-ignored)
├── src/
│   ├── evaluators/   # One evaluator per file: custom, ragas, deepeval
│   ├── pipeline/     # System under test — the CRM responder
│   ├── reporting/    # Terminal + JSON report generation
│   ├── scoring/      # Threshold logic and release gate
│   └── utils/        # Shared helpers: config_loader, data_loader
├── tests/            # pytest test files only
└── playground.py     # Root-level sandbox entry point
```

Rules:
- No nesting beyond one level inside `src/`
- No code inside `config/`, `data/`, `docs/`, `reports/`
- No data files or configs inside `src/`
- `playground.py` stays at the root — it is the user-facing entry point

---

## Python Naming

| Kind | Convention | Examples |
|---|---|---|
| Modules (files) | `snake_case` | `config_loader`, `ragas_evaluator` |
| Functions | `snake_case` | `load_config()`, `evaluate()`, `check_thresholds()` |
| Private helpers | `_snake_case` (single underscore prefix) | `_build_reply_prompt()`, `_parse_ticket_json()` |
| Classes | `PascalCase` | `OpenAICompatibleJudge` |
| Constants | `UPPER_SNAKE_CASE` | `SYSTEM_PROMPT`, `DATA_DIR`, `INVERTED_METRICS` |
| Variables | `snake_case` | `pipeline_output`, `case_id`, `llm_cfg` |

---

## Public API Pattern

Every evaluator module exposes exactly one public function:

```python
def evaluate(pipeline_output: dict, config: dict) -> dict:
    ...
```

This is the contract. All three evaluators (`custom_evaluator`, `ragas_evaluator`,
`deepeval_evaluator`) follow it. New evaluators must do the same.

Other module public APIs:

| Module | Public function(s) |
|---|---|
| `config_loader` | `load_config()`, `load_env()`, `get_*_config()` |
| `data_loader` | `load_test_cases()`, `get_case_by_id()` |
| `crm_responder` | `generate_response()` |
| `threshold_checker` | `check_thresholds()` |
| `release_gate` | `evaluate_gate()` |
| `report_generator` | `generate_report()`, `print_terminal_report()`, `save_json_report()` |

---

## Module Structure

Every Python file must follow this layout, in order:

```python
"""
module_name.py
--------------
One-line summary of what this module does.

Longer explanation if needed. What inputs it takes, what it returns,
what it connects to.

Usage:
    from src.<package>.<module> import <public_function>

Standalone test:
    python -m src.<package>.<module>
"""

# standard library imports
# third-party imports
# internal imports

# ── Constants (if any) ────────────────────────────────────────────────────────

# ── Private helpers ───────────────────────────────────────────────────────────

# ── Public function(s) ───────────────────────────────────────────────────────

# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ...
```

Rules:
- Module docstring is mandatory — no file without one
- Standalone `__main__` block is mandatory in every `src/` module — it must run
  independently and verify the module works
- Section headers use the `# ──` separator style for visual scanning

---

## `__init__.py` Files

All `__init__.py` files are intentionally empty. They exist only to mark
directories as Python packages (enabling `from src.utils.config_loader import ...`).
Do not add code, imports, or re-exports to them.

---

## Data File Responsibilities

Each data file has one and only one responsibility:

| File | Contains | Does NOT contain |
|---|---|---|
| `data/emails.json` | Customer email inputs | Context, expected outputs, thresholds |
| `data/context.json` | Retrieved context chunks | Emails, expected outputs, thresholds |
| `data/ground_truth.json` | Expected outputs + per-case overrides | Emails, context chunks |

All three are joined by `data_loader.py`. Nothing else should read data files directly.

---

## Score Conventions

| Direction | Metric Examples | Pass Condition |
|---|---|---|
| Higher is better (↑) | faithfulness, answer_relevance, ticket_status_accuracy | `score >= threshold` |
| Lower is better (↓) | hallucination | `score <= threshold` |

Inverted metrics are declared in `threshold_checker.INVERTED_METRICS`.
When adding a new metric, declare its polarity there.

---

## Commit Style

- Commits describe *what changed and why*, not *what the code does*
- Co-authored by Claude when AI-assisted
- No `--no-verify`, no skipping hooks
