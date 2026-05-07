# Coding Standards Reference

Standards and design principles applied in this project.
Use this during coding, code review, debugging, and when introducing new libraries or patterns.

---

## Official References

| Document | URL | What it governs |
|---|---|---|
| PEP 8 — Style Guide for Python Code | https://peps.python.org/pep-0008/ | Naming, layout, whitespace, imports, comments |
| PEP 20 — The Zen of Python | https://peps.python.org/pep-0020/ | Design philosophy — run `import this` |
| PEP 257 — Docstring Conventions | https://peps.python.org/pep-0257/ | Module, class, function docstrings |
| PEP 484 — Type Hints | https://peps.python.org/pep-0484/ | How to annotate function signatures |
| PEP 526 — Variable Annotations | https://peps.python.org/pep-0526/ | How to annotate variables |
| Google Python Style Guide | https://google.github.io/styleguide/pyguide.html | Naming, imports, exceptions, docstrings, structure |
| Python `functools` docs | https://docs.python.org/3/library/functools.html | `cache`, `lru_cache`, `partial`, `wraps` |
| Python `pathlib` docs | https://docs.python.org/3/library/pathlib.html | File path handling — use instead of `os.path` |
| Python `concurrent.futures` docs | https://docs.python.org/3/library/concurrent.futures.html | ThreadPoolExecutor, timeouts, futures |
| Python `typing` docs | https://docs.python.org/3/library/typing.html | `Optional`, `Union`, `dict`, `list` type hints |
| Python `dataclasses` docs | https://docs.python.org/3/library/dataclasses.html | Structured data objects without boilerplate |
| pytest docs | https://docs.pytest.org/en/stable/ | Test discovery, fixtures, parametrize, marks |
| YAML spec | https://yaml.org/spec/1.2.2/ | Config file syntax rules |

---

## Naming — PEP 8 + Google Style Guide § 3.16

### By kind

| Kind | Convention | Example |
|---|---|---|
| Module (file) | `snake_case` | `config_loader.py`, `data_loader.py` |
| Function | `snake_case` | `load_config()`, `evaluate()` |
| Internal / private function | `_snake_case` | `_load_json()`, `_null_scores()` |
| Class | `PascalCase` | `OpenAICompatibleJudge` |
| Constant (module-level) | `UPPER_SNAKE_CASE` | `INVERTED_METRICS`, `DATA_DIR` |
| Variable | `snake_case` | `pipeline_output`, `case_id` |
| Markdown doc file | `camelCase` | `projectContext.md`, `conventions.md` |
| JSON / YAML data file | `snake_case` | `config.yaml`, `ground_truth.json` |
| Test file | `test_<module>.py` | `test_crm_responder.py` |

### Underscore system — what each form means

| Form | Meaning | Python enforces? |
|---|---|---|
| `_name` | Internal to this module — callers outside should not use it directly | No — convention only |
| `name_` | Avoids clash with a Python keyword (e.g. `class_=`) | No — convention only |
| `__name` inside a class | Triggers name mangling — Python renames to `_ClassName__name` | Yes |
| `__name__` | Python-reserved magic names (`__init__`, `__main__`, `__file__`) — never invent | Yes, by Python |

PEP 8 exact quote:
> `_single_leading_underscore`: weak "internal use" indicator.
> `from M import *` does not import objects whose names start with an underscore.

---

## Imports — PEP 8 § Imports + Google Style Guide § 2.2

- One import per line — no `import os, sys`
- Order: standard library → third-party → internal. One blank line between each group.
- Use `from x import y` when the name is used frequently and is unambiguous
- Use `import x` when the module name carries meaning at the call site (e.g. `json.loads`)
- Never use `from x import *` — pollutes the namespace and defeats `_` visibility

```python
# Standard library
import json
import time
from pathlib import Path
from functools import cache
from concurrent.futures import ThreadPoolExecutor

# Third-party
from langchain_openai import ChatOpenAI
from rich.console import Console

# Internal
from src.utils.config_loader import load_config
from src.evaluators.combined_evaluator import evaluate
```

---

## Type Hints — PEP 484 + PEP 526

Type hints are not enforced at runtime in Python but they communicate intent and enable static analysis (mypy, Pyright, IDE checks).

```python
# Function signatures
def evaluate(pipeline_output: dict, config: dict) -> dict:
def get_case_by_id(case_id: str) -> dict:
def check_thresholds(scores: dict, config: dict, overrides: dict) -> dict:

# Optional return (can be None)
from typing import Optional
def _out_of_scope_handling(reply: str, intent: str) -> Optional[float]:

# List of strings
from typing import List
def load_test_cases(case_ids: Optional[List[str]] = None) -> list:
```

Use `Optional[X]` any time a function can return `None` — makes it visible at the call site that `None` must be handled.

---

## Docstrings — PEP 257 + Google Style Guide § 3.8

### Module docstring (top of every file)
```python
"""
module_name.py
--------------
One-line summary.

Longer explanation if needed.

Usage:
    from src.<package>.<module> import <function>

Standalone test:
    python -m src.<package>.<module>
"""
```

### Function docstring
```python
def evaluate(pipeline_output: dict, config: dict) -> dict:
    """
    One-line summary of what it does.

    Args:
        pipeline_output : dict returned by crm_responder.generate_response()
        config          : full config dict from load_config()

    Returns:
        dict with one entry per metric: {metric_name: {"score": float|None, "error": str|None}}

    Raises:
        ValueError if required keys are missing from pipeline_output
    """
```

Rules:
- First line is a one-sentence summary — imperative mood ("Return", "Load", "Evaluate")
- Public functions always have a docstring
- Internal helpers (`_name`) get a short one-liner minimum
- Do not describe *what the code does line by line* — describe *what it is for*

---

## Error Handling — Google Style Guide § 2.4

- Catch specific exceptions — never bare `except:` or broad `except Exception:` unless you explicitly log and re-raise or document why
- When catching broadly for resilience (evaluator failures), always surface the error in the return value
- Do not silently discard errors — the caller deserves to know what failed

```python
# Correct — specific exception, meaningful error in return
except RateLimitError:
    return _null_scores("rate limit hit — case skipped (429)")

# Correct — broad catch, but error is surfaced
except Exception as e:
    return _null_scores(f"judge failed: {type(e).__name__}: {str(e)[:120]}")

# Wrong — error silently swallowed
except Exception:
    return {}
```

Use `ValueError` for bad input at system boundaries (wrong config value, short email body).
Use `NotImplementedError` to guard incomplete code paths so they fail loudly, not silently.

---

## Caching — `functools.cache` vs `functools.lru_cache`

Both are from Python's standard `functools` module.

| Decorator | Behaviour | When to use |
|---|---|---|
| `@cache` | Unbounded — stores every unique call forever | When you want all results cached with no eviction (Python 3.9+) |
| `@lru_cache(maxsize=N)` | Bounded — evicts least recently used entries when full | When memory is a concern and only recent results matter |
| `@lru_cache(maxsize=None)` | Unbounded — same as `@cache` but verbose | Avoid — use `@cache` instead |

Rule: if you have no eviction requirement, use `@cache`. It is shorter and more honest about what it does.

The function being decorated must take **hashable** arguments (strings, ints, tuples — not lists or dicts).

```python
from functools import cache

@cache
def _load_json(filename: str) -> dict:  # str is hashable — works correctly
    ...
```

---

## File Paths — Use `pathlib`, not `os.path`

`pathlib.Path` is the modern Python standard (PEP 428, available since 3.4). It is object-oriented, readable, and cross-platform.

```python
# Correct
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
path = DATA_DIR / "emails.json"

# Avoid
import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
```

---

## Concurrency — `concurrent.futures` for timeouts

When you need to run a blocking call with a timeout, use `ThreadPoolExecutor`.
Do not use `threading.Thread` directly — `concurrent.futures` gives you clean timeout and exception handling.

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(blocking_function, arg1, arg2)
    try:
        result = future.result(timeout=case_timeout_seconds)
    except concurrent.futures.TimeoutError:
        # handle timeout — function is still running in background but we stop waiting
        raise ValueError(f"timed out after {case_timeout_seconds}s")
```

Note: the underlying thread continues running after `TimeoutError` is raised — the executor does not kill it. For short-lived functions (LLM calls, file reads) this is acceptable. For long-running processes, use `subprocess` with `timeout` instead.

---

## Config vs Code — Project Rule

Any value a user might want to tune belongs in `config.yaml`, not in source code.

Enforced path:
```
config.yaml  →  config_loader.py getter  →  module reads from getter
```

Nothing reads `config.yaml` directly except `config_loader.py`.
If a literal appears in non-test source (timeout, threshold, delay, length, temperature), move it to config.

---

## Testing — pytest Conventions

| Convention | Rule |
|---|---|
| Test file naming | `tests/test_<module>.py` |
| Test function naming | `test_<what_it_tests>` |
| Fixtures | Defined in `tests/conftest.py` — shared across test files |
| Parametrize | Use `@pytest.mark.parametrize` for data-driven tests (all 10 cases) |
| No mocking the database / data files | Hit real JSON files — mocking data broke tests in prior work |
| Assertions | Use plain `assert` — pytest rewrites them for readable failure messages |

---

## Zen of Python — Design Checks

PEP 20. Run `import this` to see the full list. Principles applied here:

| Aphorism | Where it shows up in this project |
|---|---|
| Explicit is better than implicit | Config values named clearly; no magic defaults buried in code |
| Errors should never pass silently | Every evaluator failure returns a named error, not a silent `None` |
| Simple is better than complex | Combined evaluator: 1 call, 7 metrics — explainable in one sentence |
| There should be one obvious way to do it | All data flows through `data_loader.py`; all config through `config_loader.py` |
| Readability counts | Section headers, aligned assignments, consistent spacing |
| If the implementation is hard to explain, it is a bad idea | If you can't explain a function in one sentence, split it |

---

## What This File Is Not

- Not a replacement for `conventions.md` — that owns project-specific rules (folder structure, module layout, public API contract)
- Not exhaustive — only standards actively referenced in this codebase are listed
- Update this file when a new library, pattern, or Python feature is introduced
