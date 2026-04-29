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
| Disk | 6 GB free | ~4 GB for mistral model, ~2 GB for Python packages |
| OS | macOS / Linux / Windows (WSL2) | Ollama on Windows requires WSL2 |

### 1. Install system tools

```bash
# Verify Python version
python --version   # must be 3.10+

# Install Ollama (the local LLM runtime for the SUT)
# macOS / Linux:
curl -fsSL https://ollama.com/install.sh | sh
# Windows: download installer from https://ollama.com/download

# Pull the SUT model (~4 GB download — do this once)
ollama pull mistral

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
| `ragas` | RAG quality metrics (faithfulness, answer relevance) |
| `deepeval` | LLM output quality metrics (hallucination) |
| `sentence-transformers` | Local embeddings for key facts coverage |
| `chromadb` | Vector DB (Tier 2 — installed but not active yet) |
| `rich` | Terminal output formatting |
| `python-dotenv` | API key loading from `.env` |

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
# List all 10 available test cases
python playground.py --list

# Run a single case — fastest feedback loop
python playground.py TC001

# Run specific cases
python playground.py TC003 TC008

# Run all 10 cases
python playground.py --all

# Run all 10 cases and save a JSON report to reports/
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

## How It Works

```
Test Input (email / query)
        ↓
  RAG Pipeline (simulated or real ChromaDB)
        ↓
  LLM Response — SUT (Ollama/mistral, local)
        ↓
  Evaluation Layer — Judge LLM (Gemini / Groq / Ollama, configurable)
    ├── RAGAs      → faithfulness, answer relevance
    ├── DeepEval   → hallucination
    └── Custom     → ticket status, escalation, key facts (no LLM)
        ↓
  Threshold Checker → pass / fail per metric
        ↓
  Release Gate → PASSED or FAILED
        ↓
  JSON Report → reports/
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| SUT LLM | Ollama (mistral) — local, the model being evaluated |
| Judge LLM | Groq llama-3.3-70b (default) — configurable: Groq / Gemini / Ollama |
| RAG Evaluation | RAGAs 0.4.x |
| LLM Evaluation | DeepEval |
| LLM Orchestration | LangChain |
| Config | YAML |
| Entry Point | playground.py |
| Test Runner | pytest (Tier 2) |

---

## Prerequisites

```bash
# Python 3.10+
python --version

# Ollama — needed for the SUT (CRM responder)
ollama pull mistral
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

# All 10 cases + save report
python playground.py --all --save-report

# List all available cases
python playground.py --list
```

---

## Configuration

All thresholds, metrics, and model settings live in `config/config.yaml`.
No code changes needed to tune the framework.

```yaml
llm:
  sut_model:      "mistral"                  # model being evaluated (always Ollama)
  judge_provider: "groq"                     # "groq" | "gemini" | "ollama"
  judge_model:    "llama-3.3-70b-versatile"  # model for the chosen provider

evaluation:
  mode: "combined"   # 1 LLM call per case — all 7 metrics (Tier 1 default)
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

- **Tier 1 (MVP — active):** Simulated RAG, core metrics, playground runner, JSON report
- **Tier 2:** Real ChromaDB, adversarial test cases, tone calibration, pytest suite
- **Tier 3:** Pluggable domain, CI/CD via GitHub Actions, HTML dashboard, bias testing
- **Tier 4:** Production monitoring — traffic sampling, drift detection, score trending, alerting

> Tier 3 gates releases. Tier 4 watches production after release. These are different systems — Tier 4 reuses the metric and scoring layer from Tier 1 but adds logging, scheduling, and alerting infrastructure.

---

## License

MIT
