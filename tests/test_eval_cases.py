"""
test_eval_cases.py
------------------
End-to-end integration tests for all 16 BSFI CRM test cases.

Each test case runs the full pipeline:
    1. Email → CRM responder (Ollama SUT) → generated reply
    2. Custom evaluator (deterministic) → ticket_status, escalation, key_facts, etc.
    3. Combined LLM evaluator (Groq judge) → faithfulness, hallucination, etc.
    4. Threshold checker → pass/fail per metric
    5. Assert: all critical metrics must pass their configured thresholds

Why this file exists alongside playground.py:
    playground.py  — developer sandbox, rich terminal output, interactive use
    test_eval_cases.py — formal pytest suite, CI/CD gate, pytest-html reportable

Markers:
    integration — requires Ollama running locally and Groq API key in .env

Run all integration tests with HTML report:
    pytest tests/test_eval_cases.py -v --html=reports/report.html --self-contained-html

Skip integration tests (no Ollama needed):
    pytest tests/ -m "not integration" -v

Test case coverage:
    TC001 — Loan eligibility query
    TC002 — EMI payment failure
    TC003 — Interest rate grievance (escalation)
    TC004 — Loan foreclosure request
    TC005 — KYC address update
    TC006 — Business loan status check
    TC007 — Financial hardship / moratorium (escalation)
    TC008 — Processing fee complaint (escalation)
    TC009 — Nominee addition
    TC010 — Out-of-scope + loan offer (dual intent)
    TC011 — Duplicate EMI deduction (escalation)
    TC012 — Guaranteed returns query (out-of-scope)
    TC013 — Gate 2: empty context behaviour
    TC014 — Non-English email (language routing)
    TC015 — PII leakage prevention
    TC016 — Vague/ambiguous loan query
"""

import pytest
from playground import run_case
from src.utils.data_loader import get_case_by_id
from src.utils.config_loader import load_config
from src.evaluators.combined_evaluator import LLM_METRICS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    return load_config()


# ── Case definitions ──────────────────────────────────────────────────────────

ALL_CASES = [
    ("TC001", "loan eligibility query"),
    ("TC002", "EMI payment failure"),
    ("TC003", "interest rate grievance — escalation"),
    ("TC004", "loan foreclosure request"),
    ("TC005", "KYC address update"),
    ("TC006", "business loan status check"),
    ("TC007", "financial hardship moratorium — escalation"),
    ("TC008", "processing fee complaint — escalation"),
    ("TC009", "nominee addition"),
    ("TC010", "out-of-scope + in-scope dual intent"),
    ("TC011", "duplicate EMI deduction — escalation"),
    ("TC012", "guaranteed returns query — out-of-scope"),
    ("TC013", "Gate 2 empty context behaviour"),
    ("TC014", "non-English email language routing"),
    ("TC015", "PII leakage prevention"),
    ("TC016", "vague ambiguous loan query"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _critical_failures(threshold_result: dict) -> list[str]:
    """
    Return failure strings for all critical metrics that did not pass.
    score=None (Gate 2 skipped) is not counted as a failure — it is expected
    behaviour for cases like TC013 where context is intentionally empty.
    """
    failures = []
    for metric, data in threshold_result.items():
        if metric == "id":
            continue
        if data.get("critical") and data.get("score") is not None:
            if data.get("passed") is False:
                failures.append(
                    f"{metric}: score={data['score']:.3f} threshold={data['threshold']:.2f}"
                )
    return failures


# ── Parametrized end-to-end tests ─────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("case_id,intent", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_critical_metrics_pass(case_id: str, intent: str, config: dict):
    """
    Full pipeline test — all critical metrics must pass their thresholds.

    Covers: faithfulness, answer_relevance, hallucination, answer_correctness,
            ticket_status_accuracy, escalation_logic, out_of_scope_handling.

    TC013 is expected to have LLM scores=None (Gate 2 fires — empty context).
    That is correct behaviour and does not fail this test.
    """
    test_case        = get_case_by_id(case_id)
    result           = run_case(test_case, config, verbose=False)
    threshold_result = result["threshold_result"]

    reply = result["pipeline_output"].get("generated_reply", "")
    assert isinstance(reply, str) and len(reply) > 0, (
        f"{case_id}: generated_reply is empty — SUT produced no output"
    )

    failures = _critical_failures(threshold_result)
    assert not failures, (
        f"{case_id} [{intent}] — {len(failures)} critical metric(s) failed:\n"
        + "\n".join(f"  ✗ {f}" for f in failures)
    )


@pytest.mark.integration
@pytest.mark.parametrize("case_id,intent", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_ticket_status_correct(case_id: str, intent: str, config: dict):
    """
    Ticket status predicted by SUT must match expected status in ground truth.

    This is a deterministic custom metric — always runs, score is never None.
    Ticket routing is a hard business requirement in BSFI CRM.
    """
    test_case        = get_case_by_id(case_id)
    result           = run_case(test_case, config, verbose=False)
    pipeline_output  = result["pipeline_output"]
    threshold_result = result["threshold_result"]

    status_data = threshold_result.get("ticket_status_accuracy", {})
    score       = status_data.get("score")

    assert score is not None, (
        f"{case_id}: ticket_status_accuracy score is None — custom evaluator did not run"
    )
    assert score == 1.0, (
        f"{case_id} [{intent}]: ticket status mismatch — "
        f"predicted='{pipeline_output.get('predicted_ticket_status')}' "
        f"expected='{pipeline_output['ground_truth'].get('expected_ticket_status')}'"
    )


@pytest.mark.integration
@pytest.mark.parametrize("case_id,intent", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_escalation_correct(case_id: str, intent: str, config: dict):
    """
    Escalation flag predicted by SUT must match expected escalation in ground truth.

    A false negative (missed escalation) leaves a distressed customer unattended.
    This test isolates escalation logic from the combined metric bundle.
    """
    test_case        = get_case_by_id(case_id)
    result           = run_case(test_case, config, verbose=False)
    pipeline_output  = result["pipeline_output"]
    threshold_result = result["threshold_result"]

    esc_data = threshold_result.get("escalation_logic", {})
    score    = esc_data.get("score")

    assert score is not None, (
        f"{case_id}: escalation_logic score is None — custom evaluator did not run"
    )
    assert score == 1.0, (
        f"{case_id} [{intent}]: escalation mismatch — "
        f"predicted={pipeline_output.get('predicted_escalation')} "
        f"expected={pipeline_output['ground_truth'].get('expected_escalation')}"
    )


# ── Targeted tests for boundary and adversarial cases ─────────────────────────

@pytest.mark.integration
def test_tc013_gate2_empty_context_skips_llm_eval(config):
    """
    TC013 — Gate 2: empty retrieved context must skip LLM evaluation.

    When RAG retrieves nothing, all LLM metric scores must be None.
    Custom evaluator (ticket_status, escalation) must still produce scores.
    This validates that the pipeline degrades gracefully instead of crashing.
    """
    test_case       = get_case_by_id("TC013")
    result          = run_case(test_case, config, verbose=False)
    pipeline_output = result["pipeline_output"]
    all_scores      = result["all_scores"]

    assert len(pipeline_output.get("retrieved_context", [])) == 0, (
        "TC013: retrieved_context should be empty to trigger Gate 2"
    )

    for metric in LLM_METRICS:
        score = all_scores.get(metric, {}).get("score")
        assert score is None, (
            f"TC013: {metric} should be None when Gate 2 fires, got {score}"
        )

    assert all_scores.get("ticket_status_accuracy", {}).get("score") is not None, (
        "TC013: ticket_status_accuracy must run even when Gate 2 fires"
    )
    assert all_scores.get("escalation_logic", {}).get("score") is not None, (
        "TC013: escalation_logic must run even when Gate 2 fires"
    )


@pytest.mark.integration
def test_tc014_language_check_produces_score(config):
    """
    TC014 — Non-English email: language_check must always produce a score.

    Whether the SUT replies in English (score=1.0) or another language (score=0.0),
    the metric must not be None. A None here means the custom evaluator broke.
    """
    test_case  = get_case_by_id("TC014")
    result     = run_case(test_case, config, verbose=False)
    all_scores = result["all_scores"]

    lang_score = all_scores.get("language_check", {}).get("score")
    assert lang_score is not None, "TC014: language_check must always produce a score"
    assert lang_score in (0.0, 1.0), (
        f"TC014: language_check score must be 0.0 or 1.0, got {lang_score}"
    )


@pytest.mark.integration
def test_tc015_pii_leakage_passes(config):
    """
    TC015 — PII leakage prevention: reply must not echo Aadhaar, PAN, or account numbers.

    pii_leakage is LLM-judged. Score=1.0 means no PII leaked. This is a
    regulatory requirement for BSFI — failing this in production is a compliance breach.
    """
    test_case        = get_case_by_id("TC015")
    result           = run_case(test_case, config, verbose=False)
    threshold_result = result["threshold_result"]

    pii_data = threshold_result.get("pii_leakage", {})
    score    = pii_data.get("score")

    if score is None:
        pytest.skip("TC015: pii_leakage score is None — LLM eval was skipped (Gate 2)")

    assert pii_data.get("passed"), (
        f"TC015: PII leakage detected — score={score:.3f}, "
        f"reply may be echoing customer PII back"
    )


@pytest.mark.integration
def test_tc010_out_of_scope_handled_correctly(config):
    """
    TC010 — Dual intent: out-of-scope stock query + in-scope loan offer.

    out_of_scope_handling must pass (SUT correctly declines investment advice).
    hallucination must pass (SUT must not fabricate investment advice).
    The SUT must NOT refuse the loan offer part — that is in scope.
    """
    test_case        = get_case_by_id("TC010")
    result           = run_case(test_case, config, verbose=False)
    all_scores       = result["all_scores"]
    threshold_result = result["threshold_result"]

    oos_score = all_scores.get("out_of_scope_handling", {}).get("score")
    assert oos_score is not None, "TC010: out_of_scope_handling must produce a score"
    assert oos_score == 1.0, (
        f"TC010: out_of_scope_handling failed — score={oos_score}, "
        f"notes={all_scores.get('out_of_scope_handling', {}).get('notes', '')}"
    )

    hall_data  = threshold_result.get("hallucination", {})
    hall_score = hall_data.get("score")
    if hall_score is not None:
        assert hall_data.get("passed"), (
            f"TC010: hallucination too high — score={hall_score:.3f} "
            f"threshold={hall_data.get('threshold')}"
        )


@pytest.mark.integration
def test_tc012_no_restricted_words_in_reply(config):
    """
    TC012 — Guaranteed returns query: reply must not contain restricted phrases.

    restricted_words checks for regulatory violations — 'guaranteed', 'assured returns',
    '100% safe' etc. A BSFI CRM agent must never use these phrases.
    This is a deterministic custom metric, always runs, never None.
    """
    test_case  = get_case_by_id("TC012")
    result     = run_case(test_case, config, verbose=False)
    all_scores = result["all_scores"]

    rw_data = all_scores.get("restricted_words", {})
    score   = rw_data.get("score")

    assert score is not None, "TC012: restricted_words must always produce a score"
    assert score == 1.0, (
        f"TC012: restricted words found in reply — {rw_data.get('notes', '')}"
    )
