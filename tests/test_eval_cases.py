"""
test_eval_cases.py
------------------
End-to-end integration tests for all 16 BSFI CRM test cases.

Pipeline (executed once per TC via the session-scoped pipeline_results fixture):
    1. Email → CRM responder (SUT) → generated reply
    2. Custom evaluator (deterministic) → ticket_status, escalation, key_facts, etc.
    3. Combined LLM evaluator (Judge LLM) → faithfulness, hallucination, etc.
    4. Threshold checker → pass/fail per metric

Test structure:
    test_all_metrics[TCxxx]     — one test per TC; soft-asserts every metric in a
                                   single execution. All failures surfaced together.
    Targeted standalone tests   — TC010, TC012, TC013, TC014, TC015 edge-case behaviour.

Why soft assertions (pytest-check):
    Hard assert stops at the first failure — you see one broken metric and nothing else.
    Soft assertions accumulate all failures and report them together, giving the AI
    team the full per-metric picture in one test run.

Report commands:
    Allure (full dashboard — recommended for team sharing):
        pytest tests/test_eval_cases.py -v --alluredir=reports/allure-results
        allure serve reports/allure-results

    HTML (quick local view):
        pytest tests/test_eval_cases.py -v --html=reports/report.html --self-contained-html

    Skip integration tests (unit tests only, no SUT needed):
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
import pytest_check as check
import allure

from src.utils.data_loader import load_test_cases
from src.evaluators.combined_evaluator import LLM_METRICS


# ── Case IDs — derived from data layer, not hardcoded ─────────────────────────
# Collected at module import time. load_test_cases() reads from data/*.json —
# no LLM call here, just file I/O. IDs stay in sync with the data files.

_ALL_CASE_IDS = [c["id"] for c in load_test_cases()]


# ── Metric display helpers ────────────────────────────────────────────────────

def _score_line(metric: str, data: dict) -> str:
    """Format one metric row for failure messages and Allure attachments."""
    score     = data.get("score")
    threshold = data.get("threshold")
    passed    = data.get("passed")
    critical  = data.get("critical", False)
    notes     = data.get("notes") or data.get("error", "")

    score_str     = f"{score:.3f}"     if score     is not None else "—"
    threshold_str = f"{threshold:.2f}" if threshold is not None else "—"
    status        = "PASS" if passed else ("FAIL" if passed is False else "SKIP")
    crit_marker   = " [CRITICAL]" if critical else ""

    return f"  {status}{crit_marker:11s} {metric:<28s} score={score_str:<7} threshold={threshold_str:<6} {notes}"


def _build_score_table(threshold_result: dict) -> str:
    """Full metric table — attached to every Allure test for the AI team."""
    lines = ["", f"{'STATUS':<16} {'METRIC':<28} {'SCORE':<12} {'THRESHOLD':<10} NOTES", "-" * 90]
    for metric, data in threshold_result.items():
        if metric == "id":
            continue
        lines.append(_score_line(metric, data))
    return "\n".join(lines)


def _collect_failures(threshold_result: dict) -> list[str]:
    """
    Return formatted failure lines for all critical metrics that did not pass.
    score=None (Gate 2 skipped) is not a failure — it is expected behaviour.
    """
    return [
        _score_line(metric, data)
        for metric, data in threshold_result.items()
        if metric != "id"
        and data.get("critical")
        and data.get("score") is not None
        and data.get("passed") is False
    ]


# ── Primary parametrized test — one test per TC, all metrics ─────────────────

@pytest.mark.integration
@pytest.mark.parametrize("case_id", _ALL_CASE_IDS)
def test_all_metrics(case_id: str, pipeline_results: dict) -> None:
    """
    Full pipeline assertion for one test case — all metrics checked in one test.

    Uses pytest-check (soft assertions) so every failing metric is reported
    together rather than stopping at the first failure. The Allure report
    attaches the complete score table regardless of pass/fail outcome.

    TC013 LLM scores are expected to be None (Gate 2 fires — empty context).
    That is correct behaviour and does not fail this test.
    """
    result           = pipeline_results.get(case_id, {})
    threshold_result = result.get("threshold_result", {})
    pipeline_output  = result.get("pipeline_output")

    # Rate limit or Gate 1 skip — recorded as error in fixture, not a test failure
    if result.get("error"):
        pytest.skip(f"{case_id}: pipeline did not run — {result['error'][:120]}")

    # ── Allure metadata ───────────────────────────────────────────────────────
    allure.dynamic.title(f"{case_id} — all metrics")
    allure.dynamic.label("case_id", case_id)

    if pipeline_output:
        allure.dynamic.description(
            f"Intent   : {pipeline_output.get('intent', '—')}\n"
            f"Category : {pipeline_output.get('category', '—')}\n"
            f"Model    : {pipeline_output.get('model_used', '—')}\n"
            f"Subject  : {pipeline_output.get('email_subject', '—')}"
        )
        allure.attach(
            pipeline_output.get("generated_reply", ""),
            name="Generated Reply",
            attachment_type=allure.attachment_type.TEXT,
        )

    if threshold_result:
        allure.attach(
            _build_score_table(threshold_result),
            name="Metric Scores",
            attachment_type=allure.attachment_type.TEXT,
        )

    # ── Reply must be non-empty ───────────────────────────────────────────────
    reply = pipeline_output.get("generated_reply", "") if pipeline_output else ""
    check.is_true(
        isinstance(reply, str) and len(reply) > 0,
        msg=f"{case_id}: generated_reply is empty — SUT produced no output",
    )

    # ── Assert every critical metric ─────────────────────────────────────────
    failures = _collect_failures(threshold_result)
    check.is_false(
        bool(failures),
        msg=(
            f"\n{case_id} — {len(failures)} critical metric(s) failed:\n"
            + "\n".join(failures)
        ),
    )

    # ── Deterministic metrics must always produce a score (never None) ────────
    for metric in ("ticket_status_accuracy", "escalation_logic"):
        score = threshold_result.get(metric, {}).get("score")
        check.is_not_none(
            score,
            msg=f"{case_id}: {metric} score is None — custom evaluator did not run",
        )


# ── Targeted edge-case tests ──────────────────────────────────────────────────

@pytest.mark.integration
def test_tc013_gate2_empty_context_skips_llm_eval(tc013_simulated: dict) -> None:
    """
    TC013 — Gate 2: empty retrieved context must skip all LLM evaluations.

    When RAG retrieves nothing, every LLM metric score must be None.
    Custom evaluator (ticket_status, escalation) must still produce scores.
    Validates graceful pipeline degradation — no crash, no fabricated scores.

    Uses the tc013_simulated fixture which forces RAG mode to 'simulated' so
    the empty retrieved_chunks from context.json are used instead of ChromaDB.
    """
    pipeline_output = tc013_simulated["pipeline_output"]
    all_scores      = tc013_simulated["all_scores"]

    with allure.step("retrieved_context must be empty to trigger Gate 2"):
        assert len(pipeline_output.get("retrieved_context", [])) == 0, (
            "TC013: retrieved_context should be empty to trigger Gate 2"
        )

    with allure.step("all LLM metric scores must be None when Gate 2 fires"):
        for metric in LLM_METRICS:
            score = all_scores.get(metric, {}).get("score")
            assert score is None, (
                f"TC013: {metric} should be None when Gate 2 fires, got {score}"
            )

    with allure.step("custom evaluator metrics must still run"):
        assert all_scores.get("ticket_status_accuracy", {}).get("score") is not None, (
            "TC013: ticket_status_accuracy must run even when Gate 2 fires"
        )
        assert all_scores.get("escalation_logic", {}).get("score") is not None, (
            "TC013: escalation_logic must run even when Gate 2 fires"
        )


@pytest.mark.integration
def test_tc014_language_check_produces_score(pipeline_results: dict) -> None:
    """
    TC014 — Non-English email: language_check must always produce a binary score.

    Score 1.0 = English reply (correct).  Score 0.0 = non-English reply (routing failure).
    A None score means the custom evaluator itself broke — that is the failure being caught.
    """
    result = pipeline_results["TC014"]
    if result.get("error"):
        pytest.skip(f"TC014: pipeline did not run — {result['error'][:120]}")
    all_scores = result["all_scores"]
    lang_score = all_scores.get("language_check", {}).get("score")

    check.is_not_none(lang_score, msg="TC014: language_check must always produce a score")
    check.is_in(
        lang_score,
        (0.0, 1.0),
        msg=f"TC014: language_check score must be 0.0 or 1.0, got {lang_score}",
    )


@pytest.mark.integration
def test_tc015_pii_leakage_passes(pipeline_results: dict) -> None:
    """
    TC015 — PII leakage prevention: reply must not echo Aadhaar, PAN, or account numbers.

    pii_leakage is LLM-judged. Score=1.0 means no PII leaked.
    Regulatory requirement — failing this in production is a compliance breach.
    """
    result = pipeline_results["TC015"]
    if result.get("error"):
        pytest.skip(f"TC015: pipeline did not run — {result['error'][:120]}")
    threshold_result = result["threshold_result"]
    pii_data         = threshold_result.get("pii_leakage", {})
    score            = pii_data.get("score")

    if score is None:
        pytest.skip("TC015: pii_leakage score is None — LLM eval skipped (Gate 2)")

    assert pii_data.get("passed"), (
        f"TC015: PII leakage detected — score={score:.3f}. "
        f"Reply may be echoing customer Aadhaar, PAN, or account number."
    )


@pytest.mark.integration
def test_tc010_out_of_scope_handled_correctly(pipeline_results: dict) -> None:
    """
    TC010 — Dual intent: out-of-scope stock query + in-scope loan offer.

    SUT must correctly decline investment advice (out_of_scope_handling=1.0)
    and must not fabricate advice (hallucination passes threshold).
    The in-scope loan offer part must NOT be refused.
    """
    result = pipeline_results["TC010"]
    if result.get("error"):
        pytest.skip(f"TC010: pipeline did not run — {result['error'][:120]}")
    all_scores       = result["all_scores"]
    threshold_result = result["threshold_result"]

    oos_score = all_scores.get("out_of_scope_handling", {}).get("score")
    check.is_not_none(oos_score, msg="TC010: out_of_scope_handling must produce a score")
    check.equal(
        oos_score,
        1.0,
        msg=(
            f"TC010: out_of_scope_handling failed — score={oos_score}, "
            f"notes={all_scores.get('out_of_scope_handling', {}).get('notes', '')}"
        ),
    )

    hall_data  = threshold_result.get("hallucination", {})
    hall_score = hall_data.get("score")
    if hall_score is not None:
        check.is_true(
            hall_data.get("passed"),
            msg=(
                f"TC010: hallucination too high — score={hall_score:.3f} "
                f"threshold={hall_data.get('threshold')}"
            ),
        )


@pytest.mark.integration
def test_tc012_no_restricted_words_in_reply(pipeline_results: dict) -> None:
    """
    TC012 — Guaranteed returns query: reply must not contain restricted phrases.

    Checks for regulatory violations — 'guaranteed', 'assured returns', '100% safe', etc.
    A BSFI CRM agent must never use these phrases (RBI/SEBI mis-selling risk).
    Deterministic metric — always runs, score is never None.
    """
    result = pipeline_results["TC012"]
    if result.get("error"):
        pytest.skip(f"TC012: pipeline did not run — {result['error'][:120]}")
    all_scores = result["all_scores"]
    rw_data    = all_scores.get("restricted_words", {})
    score      = rw_data.get("score")

    check.is_not_none(score, msg="TC012: restricted_words must always produce a score")
    check.equal(
        score,
        1.0,
        msg=f"TC012: restricted words found in reply — {rw_data.get('notes', '')}",
    )
