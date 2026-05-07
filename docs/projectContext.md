# Project Context — LLM Evaluation Framework

## What This Framework Is

An LLM Evaluation Framework that tests AI-powered features across any domain.

Built to answer the question:
**"How do you know your LLM is doing the right thing?"**

It integrates:
- **RAGAs** — RAG pipeline metrics (faithfulness, answer relevance, context precision, recall)
- **DeepEval** — LLM output quality evaluation, pytest-native, assertion-based
- **Custom evaluators** — domain-specific deterministic checks (no LLM required)
- **Configurable judge LLM** — Gemini (default), Groq, or Ollama via `config.yaml`

Fully tester-configurable via `config.yaml`. No code changes needed to adjust
thresholds, toggle metrics, or switch the judge LLM provider.

---

## Architecture — 5 Layers

```
Layer 1 — TESTER CONFIGURATION
  config.yaml → domain, metrics, thresholds, judge provider

Layer 2 — TEST DATA
  emails.json + context.json + ground_truth.json → joined by data_loader.py

Layer 3 — SYSTEM UNDER TEST
  CRM auto-responder (crm_responder.py) → Ollama (local) → reply + ticket status

Layer 4 — EVALUATION
  RAGAs + DeepEval + Custom evaluators → scores per metric

Layer 5 — SCORING & REPORTING
  Threshold checker → pass/fail → release gate → JSON report
```

---

## LLM Configuration — SUT vs Judge

Two separate LLM roles. Always different models — never the same.

| Role | What it does | Config key |
|---|---|---|
| **SUT** (System Under Test) | Generates the CRM reply being evaluated | `llm.sut_model` |
| **Judge LLM** | Evaluates the SUT's output for quality | `llm.judge_provider` + `llm.judge_model` |

> Rule: SUT ≠ Judge. Self-judging produces inflated scores (model rates its own output highly).

---

### Switching the Judge — 2 lines in `config.yaml`

```yaml
# Gemini (default — fast, external)
judge_provider: "gemini"
judge_model:    "gemini-2.0-flash"

# Groq (best for batch runs — higher rate limits)
# judge_provider: "groq"
# judge_model:    "llama-3.3-70b-versatile"

# Ollama (local fallback — slow, no API cost)
# judge_provider: "ollama"
# judge_model:    "mistral"
```

API keys live in `.env` — no code changes needed to switch.

---

### Available Models & Rate Limits (Free Tier)

**Gemini (GEMINI_API_KEY)**

| Model | RPM | Notes |
|---|---|---|
| `gemini-2.0-flash` | 15 | Default judge — fast, capable |
| `gemini-2.0-flash-lite` | 30 | Higher RPM, slightly less capable |

**Groq (GROQ_API_KEY)**

| Model | RPM | RPD | Notes |
|---|---|---|---|
| `llama-3.3-70b-versatile` | 30 | 1,000 | Best quality, recommended for `--all` runs |
| `llama-3.1-8b-instant` | 30 | 14,400 | Fastest, highest daily limit |
| `qwen-qwq-32b` | 60 | 1,000 | Highest RPM on free tier |

**Ollama (no key)**

| Model | Notes |
|---|---|
| `mistral` | Local only, no quota limits, slow on low RAM |

---

### How Each Evaluator Uses the Judge

| Evaluator | Mode | Library used | Notes |
|---|---|---|---|
| `combined_evaluator` | combined | `langchain_openai.ChatOpenAI` | All providers via OpenAI-compatible endpoint |
| `ragas_evaluator` | separate | `AsyncOpenAI` + `llm_factory` | All providers via OpenAI-compatible endpoint |
| `deepeval_evaluator` | separate | `OpenAICompatibleJudge` (Groq/Ollama) | Wraps any OpenAI-compatible endpoint as DeepEval judge |

> RAGAs requires `InstructorLLM` (via `llm_factory`) — `LangchainLLMWrapper` is explicitly rejected.
> DeepEval batch API: `deepeval.evaluate([test_case], metrics)` runs all metrics in 1 API call — scores populated in-place on metric objects post-call.

---

## Hands-On Domain: BSFI CRM Auto-Responder

The framework is domain-agnostic. The included implementation uses a
**BSFI (Banking, Small Finance, Micro Lending, NBFC) CRM Auto-Responder**
as the system under test.

This domain was chosen because:
- Every evaluation layer is exercisable (RAG, LLM output, ticket logic)
- High stakes = meaningful test outcomes
- Regulatory context adds a compliance testing dimension
- Represents real-world AI deployment complexity

### CRM Pipeline Under Test

```
Incoming Customer Email
      ↓
RAG Layer → retrieves policy docs, ticket history, customer profile
      ↓
LLM Layer (Ollama/mistral) → generates reply + ticket metadata
      ↓
Outgoing Reply + Ticket Status + Escalation Flag
```

---

## Tech Stack

| Component | Tool | Notes |
|---|---|---|
| Language | Python 3.10+ | DeepEval/RAGAs native |
| SUT LLM | Ollama (mistral) | Always local — this is what we're testing |
| Judge LLM | Groq llama-3.3-70b (default) | Configurable: Groq / Gemini / Ollama via `config.yaml` |
| RAG Evaluation | RAGAs 0.4.x | faithfulness, answer_relevance, context_precision, context_recall |
| LLM Evaluation | DeepEval | hallucination, faithfulness, answer_relevancy, bias, toxicity, GEval custom |
| Combined Evaluator | LangChain + custom prompt | All 7 LLM metrics in 1 Groq call — quota-efficient |
| Semantic Similarity | bert-score + rouge-score | BERTScore (roberta-large), ROUGE-1/L — no LLM needed |
| LLM Orchestration | LangChain | OllamaLLM for SUT, ChatOpenAI for judge |
| Vector DB | ChromaDB (local) | Live mode active — `data/chroma_db/` |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | chunk_size=256 words, chunk_overlap=40 |
| Config | YAML | Human-readable, tester-editable |
| Entry Point | playground.py | Hands-on single/batch case runner |
| Test Runner | pytest | `tests/` — unit, integration, gate tests |
| Reporting | JSON + rich terminal | `reports/` folder |

---

## Folder Structure

```
llm-eval-framework/
├── README.md                        ← GitHub homepage
├── playground.py                    ← hands-on entry point — run any case instantly
├── config/
│   └── config.yaml                  ← thresholds, metrics, judge provider, model settings
├── data/
│   ├── emails.json                  ← customer email inputs (TC001–TC010)
│   ├── context.json                 ← retrieved context chunks per case (simulated RAG)
│   ├── ground_truth.json            ← expected outputs + per-case threshold overrides
│   └── policy_docs/                 ← domain policy documents (Tier 2)
├── docs/
│   ├── projectContext.md            ← this file — architecture, decisions, structure
│   ├── testingPhilosophy.md         ← methodology and thought leaders
│   ├── evaluationCoverage.md        ← all 15 metric categories
│   ├── llmSyndromes.md              ← LLM bug taxonomy
│   ├── conventions.md               ← naming rules, code structure, API patterns
│   ├── codingStandards.md           ← PEP 8, Google Style Guide, Python docs — reference during coding and review
│   └── learningLog.md               ← lessons from building and debugging
├── src/
│   ├── utils/
│   │   ├── config_loader.py         ← single source of truth for all config
│   │   └── data_loader.py           ← joins emails + context + ground_truth by ID
│   ├── pipeline/
│   │   └── crm_responder.py         ← SUT: generates reply + ticket metadata
│   ├── evaluators/
│   │   ├── combined_evaluator.py    ← active: all 7 LLM metrics in 1 Groq call
│   │   ├── custom_evaluator.py      ← deterministic: ticket_status, escalation, key_facts, bert_score, rouge
│   │   └── experimental/            ← separate mode (ragas + deepeval independently)
│   │       ├── ragas_evaluator.py   ← faithfulness, answer_relevance, context_precision, recall
│   │       └── deepeval_evaluator.py← 7 core metrics + GEval (tone, OOS) + 4 RAGAs-equivalent
│   ├── scoring/
│   │   ├── threshold_checker.py     ← compare scores to thresholds
│   │   └── release_gate.py          ← go/no-go gate based on critical metrics
│   └── reporting/
│       └── report_generator.py      ← terminal rich table + JSON report file
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  ← shared fixtures (config, pipeline_output)
│   ├── test_custom_evaluator.py     ← deterministic evaluator tests (no LLM)
│   ├── test_crm_responder.py        ← SUT integration tests (requires Ollama)
│   ├── test_pipeline_gates.py       ← Gate 1 (email length) + Gate 2 (SUT validity) tests
│   ├── test_semantic_similarity.py  ← BERTScore + ROUGE tests (no LLM)
│   └── test_deepeval_metrics.py     ← DeepEval metric tests (requires_judge for Groq tests)
├── reports/                         ← generated JSON reports (git-ignored)
├── requirements.txt
└── .env                             ← API keys (git-ignored)
```

---

## Evaluator Mode — Combined vs Separate

Two modes for running LLM-based evaluation. Switched via one line in `config.yaml`.

```yaml
evaluation:
  mode: "combined"    # default — 1 LLM call per case
  # mode: "separate" # RAGAs + DeepEval independently
```

| Mode | LLM calls per case | Metrics covered | When to use |
|---|---|---|---|
| `combined` | 1 | All 7 LLM metrics in one prompt | Batch runs, rate-limited APIs, fast feedback |
| `separate` | ~10 | Same 7 metrics via RAGAs + DeepEval | Deep analysis, rigorous multi-step scoring |

**Combined covers (1 call):**
faithfulness, answer_relevance, context_precision, context_recall,
hallucination, answer_correctness, coherence

**Always deterministic (no LLM, unaffected by mode):**
ticket_status_accuracy, escalation_logic, key_facts_coverage, out_of_scope_handling

**Flow:**
```
playground.py
  → reads config.evaluation.mode
  → "combined"  → combined_evaluator.py  (1 call)
  → "separate"  → ragas_evaluator.py + deepeval_evaluator.py (~10 calls)
  → always      → custom_evaluator.py (no LLM)
```

---

## Tier Breakdown

### Tier 1 — MVP (Active)
- Simulated RAG (pre-retrieved context from context.json) + live ChromaDB (mode: "live")
- Combined evaluator: faithfulness, answer_relevance, context_precision, context_recall, hallucination, answer_correctness, coherence — 1 Groq call
- Semantic similarity: BERTScore (roberta-large) + ROUGE-1/L — no LLM
- Custom: ticket_status_accuracy, escalation_logic, key_facts_coverage, out_of_scope_handling
- Disagreement detection: flags internally inconsistent judge scores (faithfulness vs hallucination)
- Pipeline gates: Gate 1 (min email length), Gate 2 (meta_parse_error or empty context)
- playground.py runner + JSON report + release gate
- pytest suite: 5 test files covering gates, custom evaluator, SUT integration, BERTScore, DeepEval

### Tier 2 — Next
- Warning tier in release gate (critical/warning/monitor severity)
- Ground truth realignment (TC003 highest drift risk)
- Adversarial test cases (role override, PII fishing, scope boundary)
- tone_empathy as 8th metric
- Multi-intent test cases
- Cross-lingual handling
- Push to GitHub + 1-page portfolio case study

### Tier 3 — Future
- Pluggable domain (new domain via config only)
- CI/CD via GitHub Actions (automated eval on every model/prompt change)
- HTML dashboard report
- Bias and fairness testing suite
- Agentic evaluation extension

### Tier 4 — Production Monitoring
> Distinct from CI/CD gate. Tier 3 evaluates before release. Tier 4 watches what happens after.

- Production traffic sampling — log real inputs/outputs, sample for eval
- Scheduled automated scoring — nightly or rolling eval on sampled production data
- Score trend tracking — time-series store for hallucination rate, faithfulness, escalation accuracy over time
- Drift detection — alert when rolling metric average drops below baseline established at release
- Bias signal monitoring — track metric scores across customer segments (geography, loan type, language)
- Alerting — threshold breach triggers notification to AI/QA team
- Trend dashboard — visualise degradation, not just pass/fail per run

> The metric definitions, thresholds, and judge LLM integration from Tier 1 are reused directly.
> Tier 4 wraps them in a logging + scheduling + alerting infrastructure (e.g. Airflow, Evidently AI, Arize).

---

## Technical Debt — Resilience & Optimisation Backlog

Known gaps deferred from Tier 1. Must be resolved before Tier 2 batch runs and CI/CD integration.

### 1. No Timeout on Judge LLM Call
**File:** `src/evaluators/combined_evaluator.py`
**Gap:** `ChatOpenAI` is initialised with no `timeout` parameter. A hung Groq or Gemini request blocks the entire run indefinitely.
**Fix:** Set `timeout=30` on `ChatOpenAI(...)`. Catch `TimeoutError` and return `{"score": None, "error": "judge LLM timed out"}` per metric — do not retry, move to next case.

### 2. No Rate Limit Error Handling
**File:** `src/evaluators/combined_evaluator.py`
**Gap:** A `429 RateLimitError` from Groq/Gemini mid-run fails the case silently or with a raw stacktrace. No inter-case delay between back-to-back API calls on `--all` runs.
**Fix:** Catch `429` / `RateLimitError` explicitly — log the error, mark all metrics for that case as `{"score": None, "error": "rate limit hit — case skipped"}`, and continue. No retry (avoids extra cost). Add a configurable `inter_case_delay_seconds` in `config.yaml` (default 2s) to reduce RPM pressure on `--all` runs.

### 3. No Confidence Gate — LLM Called Regardless of SUT Output Quality
**File:** `playground.py`
**Gap:** The judge LLM is called even when the SUT output is invalid — empty reply, META parse error, or empty retrieved context. Evaluating a failed SUT output wastes an API call and produces meaningless scores.
**Fix:** Before calling the evaluator, check `pipeline_output["meta_parse_error"]` and `len(pipeline_output["retrieved_context"]) == 0`. If either is true, skip the LLM eval call entirely and mark all LLM metrics as `{"score": None, "error": "skipped — SUT output invalid"}`.

### 4. Embedding Encode Calls Not Batched
**File:** `src/evaluators/custom_evaluator.py`
**Gap:** `_key_facts_coverage()` calls `model.encode()` twice per case — once for reply sentences, once for key facts. Batching both into one call halves CPU time.
**Fix:** Combine sentences and facts into one list, call `model.encode()` once, then slice the result by length.

### 5. No Per-Case Timeout in Playground
**File:** `playground.py`
**Gap:** If Ollama hangs mid-generation, the entire `--all` run blocks on that one case indefinitely.
**Fix:** Wrap `generate_response()` in a `concurrent.futures.ThreadPoolExecutor` with a per-case wall-clock timeout. On timeout, record the case as failed and continue.

### 6. No Fallback if Embedding Model Fails to Load
**File:** `src/evaluators/custom_evaluator.py`
**Gap:** `_get_embedding_model()` raises an unhandled exception if the model download fails or `sentence-transformers` is missing. This crashes `key_facts_coverage` for all cases.
**Fix:** Wrap in `try/except`. On failure, return `{"score": None, "error": "embedding model unavailable — key_facts_coverage skipped"}` and continue.

### 7. data_loader Reads JSON Files on Every Call
**File:** `src/utils/data_loader.py`
**Gap:** `load_test_cases()` opens and parses all 3 JSON files on every invocation. In Tier 2 pytest parametrize this will happen once per test unless cached.
**Fix:** Add `@functools.lru_cache` on `load_test_cases()` — parsed once per process, free after that.

---

## Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| SUT LLM | Ollama (local) | This is what we're evaluating — kept local and free |
| Judge LLM | Gemini (default) | Fast, external, different model family from SUT |
| SUT ≠ Judge | Always enforced | Self-judging produces false negatives — inflated scores |
| Test data | 3 separate JSON files | Separation of concerns: emails / context / ground truth |
| RAG in Tier 1 | Simulated | Faster to MVP; real ChromaDB is Tier 2 |
| Config-driven | YAML | Testers can tune without touching code |
| Entry point | playground.py | Fast hands-on loop before committing to full pytest suite |

---

## Test Dataset Summary

10 synthetic test cases — each with input email, retrieved context,
ground truth reply, expected ticket status, escalation flag, tone,
key facts, and per-case threshold overrides.

| ID | Category | Intent |
|---|---|---|
| TC001 | loan_query | Loan eligibility |
| TC002 | emi_failure | EMI payment failure |
| TC003 | grievance | Interest rate dispute |
| TC004 | loan_closure | Foreclosure request |
| TC005 | kyc_query | KYC address update |
| TC006 | loan_query | Loan application status |
| TC007 | emi_failure | EMI restructuring / moratorium |
| TC008 | grievance | Mis-selling complaint |
| TC009 | kyc_query | Nominee update |
| TC010 | loan_query | Out-of-scope query |
