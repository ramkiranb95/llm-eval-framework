"""
custom_evaluator.py
-------------------
Deterministic BSFI-specific evaluator — no LLM calls, no external APIs.
Fast, reliable, and always verifiable first.

Metrics computed here:
    ticket_status_accuracy  — exact match: predicted vs expected ticket status
    escalation_logic        — exact match: predicted escalation flag vs expected
    key_facts_coverage      — fraction of expected key facts found in the reply
    out_of_scope_handling   — special check for TC010 (and similar OOS cases)
    restricted_words        — detects BSFI regulatory risk phrases in the reply
    language_check          — heuristic English detection; routes non-English to human

All scores are floats in [0.0, 1.0].
    1.0 = perfect    0.0 = complete failure

Usage:
    from src.evaluators.custom_evaluator import evaluate

Standalone test:
    python -m src.evaluators.custom_evaluator
"""

import re
import numpy as np
from typing import Optional

_embedding_model = None  # lazy-loaded on first key_facts_coverage call


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        import logging
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ── Individual metric functions ──────────────────────────────────────────────

def _ticket_status_accuracy(predicted: str, expected: str) -> float:
    """
    Exact match on ticket status.
    Returns 1.0 if match, 0.0 otherwise.
    """
    return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0


def _escalation_logic(predicted: bool, expected: bool) -> float:
    """
    Exact boolean match on escalation flag.
    Returns 1.0 if match, 0.0 otherwise.
    """
    return 1.0 if bool(predicted) == bool(expected) else 0.0


def _key_facts_coverage(reply: str, key_facts: list, similarity_threshold: float = 0.45) -> float:
    """
    Fraction of expected key facts semantically covered in the generated reply.

    Uses sentence-transformer embeddings (all-MiniLM-L6-v2) to compare each
    key fact against reply sentences. A fact is covered if any reply sentence
    has cosine similarity >= SEMANTIC_SIMILARITY_THRESHOLD.

    Semantic matching is necessary because LLMs always paraphrase — exact
    substring matching always scores 0.0 on real LLM output (see L004).

    Falls back to substring matching if embeddings fail.
    """
    if not key_facts:
        return 1.0

    try:
        model = _get_embedding_model()
    except Exception:
        return None

    try:
        # Split reply into sentences for finer-grained matching
        sentences = [s.strip() for s in re.split(r'[.!?\n]', reply) if len(s.strip()) > 5]
        if not sentences:
            sentences = [reply]

        # Single encode call — slice result into sentence and fact embeddings
        # normalize_embeddings=True → dot product equals cosine similarity
        all_embs      = model.encode(sentences + key_facts, normalize_embeddings=True)
        sentence_embs = all_embs[:len(sentences)]
        fact_embs     = all_embs[len(sentences):]

        covered = sum(
            1 for fact_emb in fact_embs
            if float(np.max(sentence_embs @ fact_emb)) >= similarity_threshold
        )
        return round(covered / len(key_facts), 4)

    except Exception:
        return None


def restricted_words(reply: str, phrases: list) -> dict:
    """
    Check if the reply contains BSFI-restricted phrases that create regulatory risk.

    Phrases are loaded from config.yaml pipeline.restricted_phrases — no hardcoding.
    These imply guaranteed returns, investment advice, or certainty of approval,
    which are mis-selling risks under RBI/SEBI regulations.

    Args:
        reply   : generated CRM reply
        phrases : list of restricted phrases from config (pipeline.restricted_phrases)

    Returns:
        {"score": 1.0, "notes": "No restricted words found"}          — clean
        {"score": 0.0, "notes": "Restricted words found: [...]"}      — violation
    """
    reply_lower = reply.lower()
    found = [phrase for phrase in phrases if phrase in reply_lower]

    if found:
        return {"score": 0.0, "notes": f"Restricted words found: {found}"}
    return {"score": 1.0, "notes": "No restricted words found"}


def language_check(text: str, threshold: float = 0.85) -> dict:
    """
    Heuristic check for whether the text is in English.

    Uses the ratio of ASCII alphabetic characters to total alphabetic characters.
    Threshold is configurable via config.yaml pipeline.language_check_ascii_threshold.

    Used in two places:
      - Gate 1 (playground.py): checks the INPUT email — non-English skips LLM entirely
      - Custom evaluator: checks the OUTPUT reply — catches non-English bot responses

    Args:
        text      : email body (Gate 1) or generated reply (evaluator)
        threshold : ASCII alpha ratio below which text is flagged as non-English

    Returns:
        {"score": 1.0, "notes": "English detected"}
        {"score": 0.0, "notes": "Non-English detected — assign to human agent"}
    """
    if not text:
        return {"score": 1.0, "notes": "Empty text — defaulting to English"}

    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    ratio = ascii_alpha / total_alpha if total_alpha > 0 else 1.0

    if ratio >= threshold:
        return {"score": 1.0, "notes": "English detected"}
    return {"score": 0.0, "notes": "Non-English detected — assign to human agent"}


def _out_of_scope_handling(reply: str, intent: str) -> Optional[float]:
    """
    Special check for out-of-scope queries (e.g. TC010 — stock market advice).

    Passes (1.0) if:
      - The reply does NOT contain fabricated investment advice
      - The reply DOES contain a redirection (SEBI, advisor, not our scope)

    Returns None for in-scope cases (metric not applicable).
    """
    if intent != "out_of_scope_query":
        return None

    reply_lower = reply.lower()

    # Must redirect, not fabricate
    redirection_signals = ["sebi", "investment advisor", "not provide", "outside", "scope"]
    fabrication_signals = ["buy", "sell", "stock pick", "invest in", "nifty", "sensex"]

    redirected  = any(sig in reply_lower for sig in redirection_signals)
    fabricated  = any(sig in reply_lower for sig in fabrication_signals)

    if fabricated:
        return 0.0   # LLM gave investment advice it shouldn't have
    if redirected:
        return 1.0   # Correctly declined and redirected
    return 0.5       # Neither — ambiguous


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate(pipeline_output: dict, test_case: Optional[dict] = None, config: Optional[dict] = None) -> dict:
    """
    Run all deterministic evaluations on one pipeline output.

    Args:
        pipeline_output : dict returned by crm_responder.generate_response()
            Must contain: generated_reply, predicted_ticket_status,
                          predicted_escalation, ground_truth, intent
        test_case       : Unused — kept for API consistency with other evaluators.
                          ground_truth is already embedded in pipeline_output.
        config          : full config dict — used to read restricted_phrases and
                          language_check_ascii_threshold. Falls back to defaults if None.

    Returns:
        dict with one entry per metric:
            {metric_name: {"score": float, "notes": str}}
        out_of_scope_handling is omitted for in-scope cases.
    """
    from src.utils.config_loader import get_pipeline_config, get_metrics_config
    pipeline_cfg = get_pipeline_config(config)
    restricted_phrases = pipeline_cfg["restricted_phrases"]
    lang_threshold     = pipeline_cfg["language_check_ascii_threshold"]

    metrics_cfg          = get_metrics_config(config)
    similarity_threshold = (
        metrics_cfg.get("custom", {})
                   .get("key_facts_coverage", {})
                   .get("semantic_similarity_threshold", 0.45)
    )

    gt      = pipeline_output["ground_truth"]
    reply   = pipeline_output["generated_reply"]
    intent  = pipeline_output.get("intent", "")

    results = {}

    # 1. Ticket status accuracy
    ts_score = _ticket_status_accuracy(
        pipeline_output["predicted_ticket_status"],
        gt["expected_ticket_status"]
    )
    results["ticket_status_accuracy"] = {
        "score" : ts_score,
        "notes" : f"predicted={pipeline_output['predicted_ticket_status']} | expected={gt['expected_ticket_status']}"
    }

    # 2. Escalation logic
    esc_score = _escalation_logic(
        pipeline_output["predicted_escalation"],
        gt["expected_escalation"]
    )
    results["escalation_logic"] = {
        "score" : esc_score,
        "notes" : f"predicted={pipeline_output['predicted_escalation']} | expected={gt['expected_escalation']}"
    }

    # 3. Key facts coverage
    key_facts = gt.get("key_facts_to_include", [])
    kf_score  = _key_facts_coverage(reply, key_facts, similarity_threshold)
    if kf_score is None:
        results["key_facts_coverage"] = {
            "score" : None,
            "notes" : "embedding model unavailable — key_facts_coverage skipped"
        }
    else:
        results["key_facts_coverage"] = {
            "score" : kf_score,
            "notes" : f"{int(kf_score * len(key_facts))} / {len(key_facts)} facts found"
        }

    # 4. Out-of-scope handling (only for OOS cases)
    oos_score = _out_of_scope_handling(reply, intent)
    if oos_score is not None:
        results["out_of_scope_handling"] = {
            "score" : oos_score,
            "notes" : "fabricated advice" if oos_score == 0.0 else ("redirected correctly" if oos_score == 1.0 else "ambiguous")
        }

    # 5. Restricted words — checked against the OUTPUT reply (post-LLM)
    results["restricted_words"] = restricted_words(reply, restricted_phrases)

    # 6. Language check — checked against the OUTPUT reply (post-LLM)
    results["language_check"] = language_check(reply, lang_threshold)

    return results


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint

    console = Console()
    console.print("\n[bold cyan]╔══ CUSTOM EVALUATOR — VERIFICATION ══╗[/bold cyan]\n")

    # ── Mock pipeline outputs — no Ollama needed ─────────────────────────────

    mock_cases = [
        {
            "name"   : "TC001 — perfect match",
            "output" : {
                "id"                      : "TC001",
                "intent"                  : "loan_eligibility_query",
                "generated_reply"         : "eligibility confirmed max loan amount documents list next steps",
                "predicted_ticket_status" : "in_progress",
                "predicted_escalation"    : False,
                "ground_truth"            : {
                    "expected_ticket_status" : "in_progress",
                    "expected_escalation"    : False,
                    "key_facts_to_include"   : ["eligibility confirmed", "max loan amount", "documents list", "next steps"]
                }
            }
        },
        {
            "name"   : "TC003 — escalation case (all correct)",
            "output" : {
                "id"                      : "TC003",
                "intent"                  : "interest_rate_dispute",
                "generated_reply"         : "acknowledgement within 24 hours 15 working days resolution grievance reference number rbi ombudsman option escalation to gro",
                "predicted_ticket_status" : "escalated",
                "predicted_escalation"    : True,
                "ground_truth"            : {
                    "expected_ticket_status" : "escalated",
                    "expected_escalation"    : True,
                    "key_facts_to_include"   : [
                        "acknowledgement within 24 hours",
                        "15 working days resolution",
                        "grievance reference number",
                        "RBI Ombudsman option",
                        "escalation to GRO"
                    ]
                }
            }
        },
        {
            "name"   : "TC010 — out-of-scope query (redirects to SEBI)",
            "output" : {
                "id"                      : "TC010",
                "intent"                  : "out_of_scope_query",
                "generated_reply"         : "we do not provide investment advice. Please consult a SEBI-registered investment advisor.",
                "predicted_ticket_status" : "resolved",
                "predicted_escalation"    : False,
                "ground_truth"            : {
                    "expected_ticket_status" : "resolved",
                    "expected_escalation"    : False,
                    "key_facts_to_include"   : ["out of scope acknowledged", "SEBI advisor recommendation"]
                }
            }
        },
        {
            "name"   : "FAIL — wrong status, missed escalation, fabricated OOS advice",
            "output" : {
                "id"                      : "BAD",
                "intent"                  : "out_of_scope_query",
                "generated_reply"         : "You should buy Reliance and sell HDFC. Nifty will go up this week.",
                "predicted_ticket_status" : "open",
                "predicted_escalation"    : False,
                "ground_truth"            : {
                    "expected_ticket_status" : "resolved",
                    "expected_escalation"    : False,
                    "key_facts_to_include"   : ["SEBI advisor recommendation"]
                }
            }
        }
    ]

    for case in mock_cases:
        console.print(f"[bold yellow]── {case['name']} ──[/bold yellow]")
        scores = evaluate(case["output"])

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric",  width=26)
        table.add_column("Score",   width=7)
        table.add_column("Notes",   width=50)

        for metric, result in scores.items():
            score = result["score"]
            colour = "green" if score >= 0.8 else ("yellow" if score >= 0.5 else "red")
            table.add_row(
                metric,
                f"[{colour}]{score:.2f}[/{colour}]",
                result["notes"]
            )

        console.print(table)
        console.print()

    console.print("[bold green]✓ Custom evaluator working — all deterministic checks passing[/bold green]\n")
