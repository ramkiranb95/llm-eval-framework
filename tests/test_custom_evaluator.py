"""
test_custom_evaluator.py
------------------------
Unit tests for src/evaluators/custom_evaluator.py.

All tests are deterministic — no LLM calls, no Ollama, no network.
Tests run fast and should always pass regardless of infrastructure state.

Coverage:
    _ticket_status_accuracy  — exact match, case-insensitive, mismatch
    _escalation_logic        — bool match, bool mismatch
    _key_facts_coverage      — all facts found, none found, partial, empty list
    _out_of_scope_handling   — correct redirect, fabricated advice, in-scope (None)
    evaluate()               — full function with mock outputs, and the None path
                               for key_facts_coverage when embedding is unavailable

Run:
    pytest tests/test_custom_evaluator.py -v
"""

import pytest
from src.evaluators.custom_evaluator import (
    _ticket_status_accuracy,
    _escalation_logic,
    _key_facts_coverage,
    _out_of_scope_handling,
    evaluate,
)


# ── _ticket_status_accuracy ───────────────────────────────────────────────────

class TestTicketStatusAccuracy:

    def test_exact_match_passes(self):
        assert _ticket_status_accuracy("in_progress", "in_progress") == 1.0

    def test_mismatch_fails(self):
        assert _ticket_status_accuracy("open", "in_progress") == 0.0

    def test_case_insensitive_match(self):
        assert _ticket_status_accuracy("IN_PROGRESS", "in_progress") == 1.0

    def test_all_valid_statuses_match_themselves(self):
        for status in ["open", "in_progress", "escalated", "resolved"]:
            assert _ticket_status_accuracy(status, status) == 1.0

    def test_escalated_vs_resolved_fails(self):
        assert _ticket_status_accuracy("escalated", "resolved") == 0.0

    def test_whitespace_is_stripped(self):
        assert _ticket_status_accuracy("  resolved  ", "resolved") == 1.0


# ── _escalation_logic ─────────────────────────────────────────────────────────

class TestEscalationLogic:

    def test_both_true_passes(self):
        assert _escalation_logic(True, True) == 1.0

    def test_both_false_passes(self):
        assert _escalation_logic(False, False) == 1.0

    def test_predicted_true_expected_false_fails(self):
        assert _escalation_logic(True, False) == 0.0

    def test_predicted_false_expected_true_fails(self):
        assert _escalation_logic(False, True) == 0.0

    def test_truthy_int_treated_as_true(self):
        assert _escalation_logic(1, True) == 1.0

    def test_zero_treated_as_false(self):
        assert _escalation_logic(0, False) == 1.0


# ── _key_facts_coverage ───────────────────────────────────────────────────────

class TestKeyFactsCoverage:

    def test_empty_facts_list_returns_perfect_score(self):
        assert _key_facts_coverage("any reply", []) == 1.0

    def test_all_facts_clearly_present_scores_high(self):
        reply = (
            "You are eligible for a personal loan. "
            "Please submit your salary slips and ID proof. "
            "Processing will take 5 working days."
        )
        facts = ["eligible for a personal loan", "salary slips", "ID proof"]
        score = _key_facts_coverage(reply, facts)
        assert score is not None
        assert score > 0.5, f"Expected > 0.5, got {score}"

    def test_completely_unrelated_reply_scores_low(self):
        reply = "The weather is sunny today. Have a nice day."
        facts = ["loan eligibility confirmed", "documents required", "salary slips"]
        score = _key_facts_coverage(reply, facts)
        assert score is not None
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_returns_float_in_range(self):
        reply = "Your loan application has been received."
        facts = ["loan application received"]
        score = _key_facts_coverage(reply, facts)
        if score is not None:
            assert 0.0 <= score <= 1.0

    def test_returns_none_when_embedding_model_unavailable(self, monkeypatch):
        def failing_model():
            raise RuntimeError("Model not available")
        import src.evaluators.custom_evaluator as ce
        monkeypatch.setattr(ce, "_get_embedding_model", failing_model)
        result = _key_facts_coverage("some reply", ["some fact"])
        assert result is None


# ── _out_of_scope_handling ────────────────────────────────────────────────────

class TestOutOfScopeHandling:

    def test_correct_redirect_to_sebi_passes(self):
        reply = "We do not provide investment advice. Please consult a SEBI-registered advisor."
        assert _out_of_scope_handling(reply, "out_of_scope_query") == 1.0

    def test_redirect_with_outside_keyword_passes(self):
        reply = "This is outside our scope. Please contact a qualified advisor."
        assert _out_of_scope_handling(reply, "out_of_scope_query") == 1.0

    def test_fabricated_stock_advice_fails(self):
        reply = "You should buy Reliance and sell HDFC. Nifty will go up."
        assert _out_of_scope_handling(reply, "out_of_scope_query") == 0.0

    def test_fabricated_invest_advice_fails(self):
        reply = "I recommend you invest in mutual funds and stock picks."
        assert _out_of_scope_handling(reply, "out_of_scope_query") == 0.0

    def test_in_scope_intent_returns_none(self):
        reply = "Your loan EMI is Rs. 4,500. Please check your bank account."
        assert _out_of_scope_handling(reply, "loan_eligibility_query") is None

    def test_ambiguous_reply_returns_half_score(self):
        reply = "We appreciate your query and will get back to you shortly."
        assert _out_of_scope_handling(reply, "out_of_scope_query") == 0.5


# ── evaluate() — full function ────────────────────────────────────────────────

class TestEvaluate:

    def test_in_scope_case_returns_expected_keys(self, mock_pipeline_output):
        scores = evaluate(mock_pipeline_output)
        assert "ticket_status_accuracy" in scores
        assert "escalation_logic"       in scores
        assert "key_facts_coverage"     in scores
        assert "out_of_scope_handling"  not in scores  # in-scope case — metric not applicable

    def test_oos_case_includes_oos_metric(self, mock_oos_pipeline_output):
        scores = evaluate(mock_oos_pipeline_output)
        assert "out_of_scope_handling" in scores

    def test_score_values_are_float_or_none(self, mock_pipeline_output):
        scores = evaluate(mock_pipeline_output)
        for metric, result in scores.items():
            score = result["score"]
            assert score is None or isinstance(score, float), (
                f"{metric}: expected float or None, got {type(score)}"
            )

    def test_score_range_valid(self, mock_pipeline_output):
        scores = evaluate(mock_pipeline_output)
        for metric, result in scores.items():
            score = result["score"]
            if score is not None:
                assert 0.0 <= score <= 1.0, f"{metric} score {score} out of [0, 1]"

    def test_correct_prediction_scores_1_on_deterministic_metrics(self):
        output = {
            "intent"                  : "loan_eligibility_query",
            "generated_reply"         : "Your loan application is being processed.",
            "predicted_ticket_status" : "in_progress",
            "predicted_escalation"    : False,
            "ground_truth"            : {
                "expected_ticket_status" : "in_progress",
                "expected_escalation"    : False,
                "key_facts_to_include"   : [],
            },
        }
        scores = evaluate(output)
        assert scores["ticket_status_accuracy"]["score"] == 1.0
        assert scores["escalation_logic"]["score"]       == 1.0

    def test_wrong_prediction_scores_0_on_deterministic_metrics(self):
        output = {
            "intent"                  : "interest_rate_dispute",
            "generated_reply"         : "We will look into this.",
            "predicted_ticket_status" : "open",
            "predicted_escalation"    : False,
            "ground_truth"            : {
                "expected_ticket_status" : "escalated",
                "expected_escalation"    : True,
                "key_facts_to_include"   : [],
            },
        }
        scores = evaluate(output)
        assert scores["ticket_status_accuracy"]["score"] == 0.0
        assert scores["escalation_logic"]["score"]       == 0.0

    def test_key_facts_none_when_embedding_unavailable(self, mock_pipeline_output, monkeypatch):
        import src.evaluators.custom_evaluator as ce
        monkeypatch.setattr(ce, "_get_embedding_model", lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))
        scores = evaluate(mock_pipeline_output)
        kf = scores["key_facts_coverage"]
        assert kf["score"] is None
        assert "embedding model unavailable" in kf["notes"]


# ── Parametrized across all 10 real cases ────────────────────────────────────

DETERMINISTIC_CASES = [
    ("TC001", "in_progress", False),
    ("TC002", "in_progress", False),
    ("TC003", "escalated",   True),
    ("TC004", "in_progress", False),
    ("TC005", "in_progress", False),
    ("TC006", "in_progress", False),
    ("TC007", "escalated",   True),   # EMI restructuring — regulatory concern, escalated
    ("TC008", "escalated",   True),
    ("TC009", "in_progress", False),
    ("TC010", "resolved",    False),
]


@pytest.mark.parametrize("case_id,expected_status,expected_escalation", DETERMINISTIC_CASES)
def test_ground_truth_deterministic_fields_are_consistent(case_id, expected_status, expected_escalation, all_test_cases):
    """
    Verify that ground_truth data in the JSON files matches what the test suite expects.
    These are data-integrity checks — they fail if someone edits ground_truth.json incorrectly.
    """
    case = next((c for c in all_test_cases if c["id"] == case_id), None)
    assert case is not None, f"{case_id} not found in loaded test cases"
    assert case["expected_ticket_status"] == expected_status, (
        f"{case_id}: expected status '{expected_status}', got '{case['expected_ticket_status']}'"
    )
    assert case["expected_escalation"] == expected_escalation, (
        f"{case_id}: expected escalation {expected_escalation}, got {case['expected_escalation']}"
    )
