"""
combined_evaluator.py
---------------------
Single-call LLM evaluator — scores all 7 Tier 1 LLM-based metrics in one
API call, replacing separate RAGAs and DeepEval calls.

Metrics covered (all Tier 1 LLM-based from config.yaml):
    RAGAs group   : faithfulness, answer_relevance, context_precision, context_recall
    DeepEval group: hallucination (inverted), answer_correctness, coherence

Not covered here (deterministic — no LLM needed, handled by custom_evaluator.py):
    ticket_status_accuracy, escalation_logic, key_facts_coverage, out_of_scope_handling

Why this exists:
    RAGAs and DeepEval each make multiple internal LLM calls per metric
    (claim decomposition, verification loops, synthetic question generation).
    This burns ~10 API calls per case on free-tier quota.

    This evaluator sends one structured prompt to the judge LLM and parses
    all 7 scores from a single JSON response — 1 call per case instead of ~10.

Trade-off:
    Single-pass judgment is less rigorous than RAGAs/DeepEval multi-step
    decomposition. Use combined for speed and rate-limited batch runs.
    Use separate (RAGAs + DeepEval) for deep accuracy analysis.

Switch via config.yaml:
    evaluation:
      mode: "combined"    # 1 LLM call per case — all 7 metrics
      mode: "separate"    # RAGAs + DeepEval (~10 LLM calls per case)

Usage:
    from src.evaluators.combined_evaluator import evaluate

Standalone test:
    python -m src.evaluators.combined_evaluator
"""

import json
import re
from langchain_openai import ChatOpenAI


# ── Evaluation prompt ─────────────────────────────────────────────────────────

EVAL_PROMPT = """You are an expert LLM output evaluator for a BSFI (Banking, Small Finance, Micro Lending) CRM system.
Evaluate the AI-generated reply below across 7 quality metrics.

--- CUSTOMER QUESTION ---
{user_input}

--- RETRIEVED CONTEXT (policy document chunks the reply must be based on) ---
{context}

--- AI GENERATED REPLY ---
{response}

--- EXPECTED REPLY (ground truth written by a human expert) ---
{expected_reply}

--- EVALUATION INSTRUCTIONS ---
Score each metric strictly from 0.0 to 1.0. Be precise and critical.

1. faithfulness
   What fraction of factual claims in the AI reply are directly supported by the retrieved context?
   1.0 = every claim traceable to context | 0.0 = no claims supported

2. answer_relevance
   How well does the AI reply address the customer's actual question?
   1.0 = fully and directly answers the question | 0.0 = completely off-topic

3. context_precision
   What fraction of the retrieved context chunks are actually relevant to this question?
   1.0 = all chunks are useful | 0.0 = all chunks are noise

4. context_recall
   What fraction of the information needed to produce the expected reply is present in the retrieved context?
   1.0 = context contains everything needed | 0.0 = critical information is missing

5. hallucination
   What fraction of the AI reply contains facts NOT present in the retrieved context?
   NOTE: LOWER IS BETTER for this metric.
   0.0 = no hallucination (all facts grounded in context)
   1.0 = fully hallucinated (no facts traceable to context)

6. answer_correctness
   How closely does the AI reply match the expected reply in content, key facts, and intent?
   1.0 = matches expected reply fully | 0.0 = completely wrong or missing key content

7. coherence
   Is the AI reply internally logical, well-structured, and professionally written?
   1.0 = clear, professional, well-organised | 0.0 = incoherent or poorly structured

Return ONLY a valid JSON object with no extra text before or after:
{{"faithfulness": <float>, "answer_relevance": <float>, "context_precision": <float>, "context_recall": <float>, "hallucination": <float>, "answer_correctness": <float>, "coherence": <float>, "reasoning": "<one sentence summary of overall quality>"}}"""


# ── Constants ─────────────────────────────────────────────────────────────────

LLM_METRICS = [
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "hallucination",
    "answer_correctness",
    "coherence",
]

INVERTED_METRICS = {"hallucination"}  # lower score = better


# ── Core function ─────────────────────────────────────────────────────────────

def evaluate(pipeline_output: dict, config: dict) -> dict:
    """
    Score all 7 Tier 1 LLM-based metrics in a single judge LLM call.

    Uses judge LLM configured in config.yaml (Gemini / Groq / Ollama).
    All providers route through ChatOpenAI → OpenAI-compatible endpoint.

    Args:
        pipeline_output : dict from crm_responder.generate_response()
        config          : full config dict from load_config()

    Returns:
        {
            "faithfulness"      : {"score": float|None, "error": str|None},
            "answer_relevance"  : {"score": float|None, "error": str|None},
            "context_precision" : {"score": float|None, "error": str|None},
            "context_recall"    : {"score": float|None, "error": str|None},
            "hallucination"     : {"score": float|None, "error": str|None, "reason": str},
            "answer_correctness": {"score": float|None, "error": str|None},
            "coherence"         : {"score": float|None, "error": str|None},
        }
        hallucination is inverted — lower score = less hallucination = better.
    """
    from src.utils.config_loader import get_judge_config

    judge_cfg      = get_judge_config(config)
    user_input     = pipeline_output["input_email"]
    response       = pipeline_output["generated_reply"]
    contexts       = pipeline_output["retrieved_context"]
    expected_reply = pipeline_output["ground_truth"]["expected_reply"]
    context_text   = "\n".join(f"- {c}" for c in contexts)

    prompt = EVAL_PROMPT.format(
        user_input     = user_input,
        context        = context_text,
        response       = response,
        expected_reply = expected_reply
    )

    try:
        llm = ChatOpenAI(
            model       = judge_cfg["model"],
            base_url    = judge_cfg["base_url"],
            api_key     = judge_cfg["api_key"],
            temperature = 0.0
        )
        raw = llm.invoke(prompt).content.strip()

        # Strip markdown code fences if model wraps response in ```json ... ```
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {raw[:200]}")

        parsed    = json.loads(json_match.group())
        reasoning = parsed.get("reasoning", "")

        scores = {}
        for metric in LLM_METRICS:
            result = extract_score(parsed, metric)
            if metric == "hallucination":
                result["reason"] = reasoning
            scores[metric] = result

        return scores

    except Exception as e:
        err = f"Combined judge failed: {type(e).__name__}: {str(e)[:120]}"
        scores = {}
        for metric in LLM_METRICS:
            scores[metric] = {"score": None, "error": err}
            if metric == "hallucination":
                scores[metric]["reason"] = ""
        return scores


# ── Private helpers ───────────────────────────────────────────────────────────

def extract_score(parsed: dict, key: str) -> dict:
    """Extract and validate a single float score from the parsed JSON."""
    val = parsed.get(key)
    if val is None:
        return {"score": None, "error": f"Key '{key}' missing from judge response"}
    try:
        score = float(val)
    except (TypeError, ValueError):
        return {"score": None, "error": f"Score is not a number: {val!r}"}
    if not (0.0 <= score <= 1.0):
        return {"score": None, "error": f"Score out of range [0,1]: {score}"}
    return {"score": round(score, 4), "error": None}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from src.utils.config_loader import load_config, get_metrics_config
    from src.utils.data_loader import get_case_by_id
    from src.pipeline.crm_responder import generate_response

    console = Console()
    console.print("\n[bold cyan]╔══ COMBINED EVALUATOR — VERIFICATION (TC001) ══╗[/bold cyan]\n")
    console.print("[dim]1 LLM call → 7 metrics: faithfulness, answer_relevance, context_precision,\n"
                  "             context_recall, hallucination, answer_correctness, coherence[/dim]\n")

    config    = load_config()
    test_case = get_case_by_id("TC001")

    console.print("  → Running CRM responder...")
    result = generate_response(test_case, config)

    console.print("  → Running combined evaluation (1 LLM call)...\n")
    scores      = evaluate(result, config)
    metrics_cfg = get_metrics_config(config)

    thresholds = {}
    for group in ["ragas", "deepeval"]:
        for metric, cfg in metrics_cfg[group].items():
            thresholds[metric] = cfg["threshold"]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric",    width=20)
    table.add_column("Score",     width=10)
    table.add_column("Threshold", width=10)
    table.add_column("Pass",      width=10)
    table.add_column("Note",      width=44)

    for metric, data in scores.items():
        score  = data.get("score")
        error  = data.get("error")
        reason = data.get("reason", "")
        thresh = thresholds.get(metric)

        if score is not None and thresh is not None:
            passed    = (score <= thresh) if metric in INVERTED_METRICS else (score >= thresh)
            colour    = "green" if passed else "red"
            inv       = " ↓" if metric in INVERTED_METRICS else ""
            score_str = f"[{colour}]{score:.4f}{inv}[/{colour}]"
            pass_str  = f"[{colour}]{'✓ PASS' if passed else '✗ FAIL'}[/{colour}]"
            note      = reason[:42] if reason else "—"
        else:
            score_str = "[dim]None[/dim]"
            pass_str  = "[yellow]⚠ error[/yellow]"
            note      = (error or "")[:42]

        table.add_row(metric, score_str, str(thresh) if thresh else "—", pass_str, note)

    console.print(table)
    console.print("\n[bold green]✓ Combined evaluator done — 1 API call, 7 metrics[/bold green]\n")
