# LLM Evaluation Framework

A configurable, open-source framework for evaluating LLM-powered features —
RAG pipelines, auto-responders, and AI agents.

Built for QA engineers, AI testers, and teams who need systematic,
evidence-based quality evaluation of LLM outputs.

> See [docs/projectContext.md](docs/projectContext.md) for full architecture,
> decisions, and folder structure.

---

## Quick Start

### System Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | DeepEval and RAGAs require 3.10 or higher |
| RAM | 8 GB | Mistral (4B) needs ~5 GB free to run in Ollama |
| Disk | 6 GB free | ~4 GB for a typical SUT model, ~2 GB for Python packages |
| OS | macOS / Linux / Windows (WSL2) | Ollama on Windows requires WSL2 |

### 1. Install system tools

```bash
# Verify Python version
python --version   # must be 3.10+

# Install Ollama (the local LLM runtime for the SUT)
# macOS / Linux:
curl -fsSL https://ollama.com/install.sh | sh
# Windows: download installer from https://ollama.com/download

# Pull the SUT model — whichever model you set as sut_model in config.yaml
# e.g. ollama pull mistral  or  ollama pull llama3.2:3b
ollama pull <your-sut-model>

# Start Ollama server (keep this running in a separate terminal)
ollama serve
```

### 2. Clone and install Python dependencies

```bash
git clone https://github.com/ramkiranb95/llm-eval-framework.git
cd llm-eval-framework

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Key packages installed** (from `requirements.txt`):

| Package | Purpose |
|---|---|
| `langchain`, `langchain-ollama`, `langchain-openai` | LLM orchestration |
| `ragas` | RAG quality metrics (faithfulness, answer relevance, context precision/recall) |
| `deepeval` | LLM evaluation — hallucination, faithfulness, GEval custom criteria |
| `sentence-transformers` | Local embeddings — ChromaDB indexing + key facts coverage |
| `rouge-score` | Lexical overlap — ROUGE-1, ROUGE-2, ROUGE-L — pure Python, no model |
| `chromadb` | Vector DB — live RAG mode (`rag.mode: "live"` in config.yaml) |
| `rich` | Terminal output formatting |
| `python-dotenv` | API key loading from `.env` |
| `pytest-check` | Soft assertions — accumulate all metric failures per test case before reporting |
| `allure-pytest` | Structured HTML test reports with per-case metric score attachments |

### 3. Configure your judge LLM API key

The **SUT** (model being evaluated) always runs locally via Ollama — no API key needed.
The **Judge LLM** evaluates the SUT's output and requires an API key.

```bash
# Create your .env file
cp .env.example .env   # or create it manually
```

Add one of the following to `.env`:

```bash
# Option A — Groq (recommended, free tier: 30 RPM / 1000 requests/day)
# Get your key at: https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Option B — Gemini (requires billing-enabled Google account)
# Get your key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Option C — No key (uses local Ollama as judge — slow, no quota)
# No .env entry needed. Set judge_provider: "ollama" in config/config.yaml
```

Then confirm your judge provider in `config/config.yaml`:

```yaml
llm:
  judge_provider: "groq"                    # "gemini" | "groq" | "ollama"
  judge_model:    "llama-3.3-70b-versatile" # model for the chosen provider
```

### 4. Run

```bash
# List all available test cases
python playground.py --list

# Run a single case — fastest feedback loop
python playground.py TC001

# Run specific cases
python playground.py TC003 TC008

# Run all cases
python playground.py --all

# Run all cases and save a JSON report to reports/
python playground.py --all --save-report
```

Scores, pass/fail per metric, and the release gate verdict print immediately after each case.

---

## What It Does

Runs automated evaluation of LLM responses across multiple quality dimensions:

- **RAG quality** — Is the retrieved context being used faithfully?
- **Output quality** — Is the reply accurate, relevant, and grounded?
- **Domain correctness** — Does the response follow business rules?
- **Release gate** — Should this LLM output be trusted in production?

---

## How It Works — E2E Pipeline

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 0 — DATA LOADING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  data_loader.py joins 3 files by test case ID:
    emails.json        → customer email, subject, intent, priority
    context.json       → pre-retrieved policy chunks (simulated RAG)
    ground_truth.json  → expected reply, ticket status, key facts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 1 — GATE 1: PRE-LLM CHECKS (input validation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Runs BEFORE any LLM call — saves tokens if input is invalid.

  1a. Length check
      email body < min_email_body_length (config.yaml)
      → raises ValueError, case skipped

  1b. Language check (input)
      ASCII alpha ratio < language_check_ascii_threshold (config.yaml)
      → non-English detected → case skipped, route to human agent

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 2 — RAG RETRIEVAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Controlled by rag.mode in config.yaml:

  simulated → context chunks read directly from context.json
              (deterministic, stable for regression testing)

  live      → email body embedded via all-MiniLM-L6-v2
              → top-k chunks retrieved from ChromaDB (data/chroma_db/)
              → build index once: python -m src.rag.retriever

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 3 — SUT: CRM RESPONDER (model under test)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  crm_responder.py sends ONE prompt to the SUT containing:
    - SYSTEM prompt (CRM agent persona)
    - Retrieved context chunks
    - Customer email

  SUT returns in a single response (parsed via regex):
    [REPLY]     → generated CRM reply
    [META]      → ticket_status + escalation_flag + reasoning (JSON)

  meta_parse_error = True if [META] block is missing or malformed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 4 — CUSTOM EVALUATOR (deterministic, no LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Always runs — fast, no API cost.

  ticket_status_accuracy  exact match: predicted vs expected status
  escalation_logic        exact match: predicted vs expected escalation flag
  key_facts_coverage      semantic similarity — fraction of ground truth
                          key facts covered in the reply
                          (threshold from config: semantic_similarity_threshold)
  out_of_scope_handling   checks OOS cases redirect correctly (TC010 etc.)
  restricted_words        detects RBI/SEBI-prohibited phrases in the reply
                          (phrase list from config: pipeline.restricted_phrases)
  language_check (output) ASCII ratio check on the generated reply
                          (catches non-English bot responses)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 5 — GATE 2: CONFIDENCE GATE (SUT output validity)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Runs AFTER custom eval, BEFORE judge LLM — saves API quota.

  Fires if:
    meta_parse_error = True  (SUT failed to produce structured output)
    retrieved_context = []   (empty context — nothing to evaluate against)

  → All 15 LLM metrics marked as null (score: None, error: "skipped")
  → Pipeline continues to threshold check and release gate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 6 — COMBINED LLM EVALUATOR (judge LLM, 1 API call)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Single structured prompt sent to the judge LLM.
  Judge runs at temperature=0 — deterministic, reproducible scores.
  Returns all 15 scores + per-metric reasoning in one JSON response.

  RAGAs group      faithfulness, answer_relevance,
                   context_precision, context_recall
  DeepEval group   hallucination (inverted), answer_correctness, coherence
  BSFI-specific    tone_professionalism, toxicity (inverted),
                   non_advice, topic_adherence, bias,
                   pii_leakage, role_adherence, answer_similarity

  Disagreement detection: if faithfulness is high but hallucination is also
  high, the judge is contradicting itself — both scores are flagged with a
  disagreement_warning (does not change the score, adds transparency).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 7 — THRESHOLD CHECKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Compares every metric score against its configured threshold.
  All thresholds from config.yaml — no per-case exceptions.

  Normal metrics   pass if score >= threshold
  Inverted metrics pass if score <= threshold  (hallucination, toxicity)

  Output per metric: score | threshold | passed | critical | inverted

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 8 — ROUTING DECISION LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  After all scores are available, logs the recommended action:
    AUTO-RESPOND  all critical metrics pass
    HUMAN REVIEW  any critical metric fails or score = None
  Reasons listed per failing metric.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 9 — RELEASE GATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Runs after ALL test cases complete (batch level).
  Policy: all_critical_must_pass (config: release_gate.policy)

  If ANY critical metric fails across ANY test case → GATE FAILS
  Non-critical failures do not block the gate.

  Output:
    RELEASE GATE PASSED — all critical metrics within threshold
    RELEASE GATE FAILED — one or more critical metrics below threshold
                          + list of failing cases and metrics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 10 — REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Terminal: rich-formatted table per case + gate banner
  JSON:     reports/run_<timestamp>.json (--save-report flag)
            includes all scores, thresholds, pass/fail,
            routing decisions, and gate result per case
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| SUT LLM | Configurable via config.yaml (sut_provider + sut_model) — the model being evaluated |
| Judge LLM | Configurable via config.yaml (judge_provider + judge_model) — evaluates SUT output |
| Combined Evaluator | LangChain + custom prompt — all 15 LLM metrics in 1 API call — framework-sourced definitions |
| RAG Evaluation | RAGAs 0.4.x — faithfulness, answer_relevance, context_precision, context_recall |
| LLM Evaluation | DeepEval — hallucination, coherence, tone, toxicity, bias, and more |
| Vector DB | ChromaDB (local) — live mode active |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) — chunk_size=256 words |
| LLM Orchestration | LangChain |
| Config | YAML |
| Entry Point | playground.py |
| Test Runner | pytest — covering gates, evaluators, SUT integration |

---

## Prerequisites

```bash
# Python 3.10+
python --version

# SUT LLM — pull whichever model you configure as sut_model in config.yaml
# e.g. ollama pull mistral  or  ollama pull llama3.2:3b
ollama pull <your-sut-model>
ollama serve

# Judge LLM — add API key to .env (Gemini is default)
# GEMINI_API_KEY=your_key_here   → judge_provider: "gemini"
# GROQ_API_KEY=your_key_here     → judge_provider: "groq"
# No key needed                  → judge_provider: "ollama" (slow, local)
```

---

## Setup

```bash
git clone https://github.com/ramkiranb95/llm-eval-framework.git
cd llm-eval-framework

python -m venv .venv
source .venv/bin/activate      # macOS/Linux

pip install -r requirements.txt

cp .env.example .env
# Add GEMINI_API_KEY or GROQ_API_KEY to .env
```

---

## Run

```bash
# Single case — fast feedback loop
python playground.py TC001

# Multiple cases
python playground.py TC003 TC008

# All cases + save report
python playground.py --all --save-report

# List all available cases
python playground.py --list
```

```bash
# Run all 21 evaluation tests (requires SUT + Judge API keys)
pytest tests/test_eval_cases.py -v

# Run with Allure report generation
pytest tests/test_eval_cases.py --alluredir=reports/allure-results
allure serve reports/allure-results

# Run only deterministic tests — zero external dependencies
pytest tests/test_custom_evaluator.py tests/test_pipeline_gates.py tests/test_semantic_similarity.py -v

# Run SUT integration tests (requires SUT provider running)
pytest tests/test_crm_responder.py -v -m integration
```

---

## Configuration

All thresholds, metrics, and model settings live in `config/config.yaml`.
No code changes needed to tune the framework.

```yaml
llm:
  sut_provider:   "ollama"                   # "ollama" | "groq" | "gemini" | "cerebras"
  sut_model:      "<your-sut-model>"           # model being evaluated
  judge_provider: "groq"                     # "groq" | "gemini" | "ollama"
  judge_model:    "<your-judge-model>"         # model for the chosen provider

evaluation:
  mode: "combined"   # 1 LLM call — all 15 metrics (default, use for pytest + batch runs)
  # mode: "separate" # RAGAs + DeepEval libraries (~12 calls) — deep analysis, single-case only
  ragas:
    faithfulness:
      enabled: true
      threshold: 0.75
      critical: true
  deepeval:
    hallucination:
      enabled: true
      threshold: 0.20    # inverted — lower is better
      critical: true
```

---

## Docs

| File | Purpose |
|---|---|
| [docs/projectContext.md](docs/projectContext.md) | Architecture, LLM config, folder structure, decisions |
| [docs/testingPhilosophy.md](docs/testingPhilosophy.md) | Bach, Bolton, Parwal, Pyhäjärvi |
| [docs/evaluationCoverage.md](docs/evaluationCoverage.md) | All 15 metric categories across tiers |
| [docs/llmSyndromes.md](docs/llmSyndromes.md) | LLM bug taxonomy |
| [docs/conventions.md](docs/conventions.md) | Naming rules, code structure, API patterns |
| [docs/learningLog.md](docs/learningLog.md) | Lessons from building and debugging |

---

## Roadmap

- **Tier 1 (active):** Simulated + live ChromaDB RAG, combined evaluator (15 LLM metrics, 1 API call), pipeline guardrails (Gate 1 + Gate 2), disagreement detection, pytest suite
- **Tier 2:** Warning tier in release gate, adversarial test cases, tone_empathy metric, ground truth realignment, push to GitHub
- **Tier 3:** Pluggable domain (new domain via config), CI/CD via GitHub Actions, HTML dashboard, bias testing suite
- **Tier 4:** Production monitoring — traffic sampling, drift detection, score trending, alerting

> Tier 3 gates releases. Tier 4 watches production after release. These are different systems — Tier 4 reuses the metric and scoring layer from Tier 1 but adds logging, scheduling, and alerting infrastructure.

---

## License

MIT
