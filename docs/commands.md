# Commands — LLM Evaluation Framework

> How to run, filter, and debug the evaluation pipeline. Config switching, report generation, and standalone module verification.

---

## Table of Contents

1. [Playground — Primary Entry Point](#1-playground--primary-entry-point)
2. [pytest — Secondary Validation](#2-pytest--secondary-validation)
3. [Allure Report](#3-allure-report)
4. [Standalone Module Verification](#4-standalone-module-verification)
5. [Config Switching](#5-config-switching)
6. [RAG Setup](#6-rag-setup)
7. [Debug and Inspection](#7-debug-and-inspection)

---

## 1. Playground — Primary Entry Point

`playground.py` is the primary validation tool. Fix issues here before running pytest.

### Run a Single Case

```bash
python playground.py --case TC001
```

### Run All 16 Cases

```bash
python playground.py --all
```

### Run Specific Cases

```bash
python playground.py --cases TC001 TC003 TC013
```

### Silent Mode (no terminal output — scores only)

```bash
python playground.py --case TC001 --no-verbose
```

---

## 2. pytest — Secondary Validation

Run only after playground passes. pytest validates the same pipeline under structured assertions with Allure reporting.

### Run All 21 Tests

```bash
pytest tests/test_eval_cases.py -v --alluredir=reports/allure-results
```

### Run a Single Case

```bash
pytest tests/test_eval_cases.py -k "TC001" -v --alluredir=reports/allure-results
```

### Run Multiple Cases

```bash
pytest tests/test_eval_cases.py -k "TC001 or TC002 or TC003" -v --alluredir=reports/allure-results
```

### Run with Log Capture (save output to file)

```bash
pytest tests/test_eval_cases.py -v --alluredir=reports/allure-results 2>&1 | tee reports/pytest_run.log
```

### Run Unit Tests Only (no LLM calls)

```bash
pytest tests/test_config_loader.py tests/test_data_loader.py tests/test_custom_evaluator.py -v
```

### Run with Short Traceback (cleaner output)

```bash
pytest tests/test_eval_cases.py -v --tb=short --alluredir=reports/allure-results
```

### Stop After First Failure

```bash
pytest tests/test_eval_cases.py -v -x --alluredir=reports/allure-results
```

---

## 3. Allure Report

### Serve Interactive Report (opens in browser)

```bash
allure serve reports/allure-results
```

### Generate Static HTML Report

```bash
allure generate reports/allure-results -o reports/allure-html --clean
```

### Open Static Report

```bash
allure open reports/allure-html
```

### Install Allure CLI (one-time)

```bash
brew install allure
```

---

## 4. Standalone Module Verification

Every `src/` module has a `__main__` block. Run these to verify a module in isolation before running the full suite.

### Config Loader

```bash
python -m src.utils.config_loader
```

### Data Loader

```bash
python -m src.utils.data_loader
```

### CRM Responder (SUT — generates one reply)

```bash
python -m src.pipeline.crm_responder
```

### RAG Retriever

```bash
python -m src.pipeline.rag_retriever
```

### Combined Evaluator (Tier 1 — 1 judge call)

```bash
python -m src.evaluators.combined_evaluator
```

### Custom Evaluator (no LLM — deterministic checks)

```bash
python -m src.evaluators.custom_evaluator
```

### RAGAs Evaluator (Tier 2 — multi-call, uses judge LLM)

```bash
python -m src.evaluators.experimental.ragas_evaluator
```

### DeepEval Evaluator (Tier 2 — batch call, uses judge LLM)

```bash
python -m src.evaluators.experimental.deepeval_evaluator
```

### Threshold Checker

```bash
python -m src.scoring.threshold_checker
```

### Release Gate

```bash
python -m src.scoring.release_gate
```

### Report Generator

```bash
python -m src.reporting.report_generator
```

---

## 5. Config Switching

All provider switching is done in `config/config.yaml`. Nothing else needs to change.

### SUT Provider Options

```yaml
# Cerebras — fast, 14.4K RPD, 60K TPM free
sut_provider: "cerebras"
sut_model: "llama3.1-8b"

# Ollama — local, no quota, slower
sut_provider: "ollama"
sut_model: "llama3.2:3b"

# Gemini — reliable structured output
sut_provider: "gemini"
sut_model: "gemini-2.0-flash-lite"
```

### Judge LLM Options

```yaml
# Gemini — best for separate mode (paid: 2K RPM, unlimited RPD)
judge_provider: "gemini"
judge_model: "gemini-2.0-flash"

# Gemini free — sufficient for combined mode or 1-3 case separate mode
judge_provider: "gemini"
judge_model: "gemini-2.0-flash-lite"

# Ollama — local, no quota, too slow for RAGAs multi-call on small models
judge_provider: "ollama"
judge_model: "mistral"
```

### Evaluation Mode

```yaml
# Tier 1 — 1 judge call per case, all 15 metrics, fast
evaluation:
  mode: "combined"

# Tier 2 — actual RAGAs + DeepEval libraries, ~20 judge calls per case
evaluation:
  mode: "separate"
```

### RAG Mode

```yaml
# Live — queries ChromaDB (requires ingestion to have run)
rag:
  mode: "live"

# Simulated — reads from data/context.json (no ChromaDB needed)
rag:
  mode: "simulated"
```

### Verify Config After Switching

```bash
python -m src.utils.config_loader
```

---

## 6. RAG Setup

Run once before first `live` mode run, or after policy docs change.

### Ingest Policy Documents into ChromaDB

```bash
python -m src.pipeline.rag_retriever --ingest
```

### Verify ChromaDB Has Data

```bash
python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/chroma_db/')
col = client.get_collection('policy_docs')
print('Documents in ChromaDB:', col.count())
"
```

---

## 7. Debug and Inspection

### Check Gemini Quota Usage

Visit: `https://ai.dev/rate-limit`

### Check Cerebras Quota Usage

Visit: `https://cloud.cerebras.ai/platform` → Analytics tab

### Print a Specific Test Case

```bash
python -c "
from src.utils.data_loader import get_case_by_id
import json
tc = get_case_by_id('TC001')
print(json.dumps(tc, indent=2))
"
```

### Print Ground Truth for a Case

```bash
python -c "
from src.utils.data_loader import get_case_by_id
tc = get_case_by_id('TC003')
gt = tc.get('ground_truth', {})
print('Expected status:', gt.get('expected_ticket_status'))
print('Expected escalation:', gt.get('expected_escalation'))
print('Expected reply:', gt.get('expected_reply', '')[:200])
"
```

### List All Test Case IDs

```bash
python -c "
from src.utils.data_loader import load_test_cases
cases = load_test_cases()
print([c['id'] for c in cases])
"
```

### Check Ollama Is Running and Models Available

```bash
curl http://localhost:11434/api/tags | python3 -c "
import sys, json
models = json.load(sys.stdin).get('models', [])
for m in models:
    print(m['name'], m.get('size', 0)//1_000_000, 'MB')
"
```

### Run Config Validation Only (no LLM calls)

```bash
python -c "
from src.utils.config_loader import load_config
load_config()
print('Config valid')
"
```

### Inspect Latest JSON Report

```bash
python -c "
import json, glob, os
reports = sorted(glob.glob('reports/run_*.json'))
if reports:
    latest = reports[-1]
    print('Latest:', latest)
    data = json.load(open(latest))
    for case in data.get('cases', []):
        print(case['id'], '—', 'PASS' if case.get('release_gate_passed') else 'FAIL')
"
```
