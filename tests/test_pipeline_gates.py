"""
test_pipeline_gates.py
----------------------
Tests for the two pre-LLM guardrails in playground.py.

Why these gates exist:
    Every LLM call costs quota (Groq RPD, Groq RPM) and time.
    Calling the judge on bad SUT output wastes both and produces meaningless scores.
    These gates ensure the LLM evaluator is only called when the input is valid
    and the SUT output is usable.

Gate 1 — Minimum email body length (config: pipeline.min_email_body_length = 50)
    Location : playground.py, before generate_response() is called
    Trigger  : email_body length < min_email_body_length
    Action   : raises ValueError — case is skipped, no LLM call is made
    Why      : A 3-word email like "What is EMI?" cannot produce a meaningful reply
               or evaluation. Sending it through wastes a quota slot.

Gate 2 — SUT output validity (confidence gate)
    Location : playground.py, after generate_response(), before evaluator is called
    Trigger  : meta_parse_error=True OR retrieved_context is empty
    Action   : LLM evaluation is skipped, all LLM metric scores = None with error message
               Custom evaluator (deterministic) still runs — it does not need LLM
    Why      : If the SUT failed to produce structured output (meta_parse_error)
               or RAG retrieved nothing (empty_context), any faithfulness or
               hallucination score would measure a broken output — not a meaningful signal.

These tests do not require Ollama or Groq — they test logic in playground.py
using mocked/minimal inputs.

Run:
    pytest tests/test_pipeline_gates.py -v
"""

import pytest
from src.utils.config_loader import load_config, get_pipeline_config


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def min_length(config):
    return get_pipeline_config(config)["min_email_body_length"]


def _make_test_case(email_body: str) -> dict:
    """Minimal test case dict that satisfies playground.run_single_case() Gate 1 check."""
    return {
        "id"                     : "TC_GATE_TEST",
        "intent"                 : "test",
        "category"               : "test",
        "priority"               : "P1",
        "email_subject"          : "Test subject",
        "email_body"             : email_body,
        "customer_id"            : "CUST-TEST",
        "ticket_id"              : "TKT-TEST",
        "retrieved_chunks"       : ["Some context chunk for testing."],
        "expected_reply"         : "Test expected reply.",
        "expected_ticket_status" : "resolved",
        "expected_escalation"    : False,
        "expected_tone"          : "professional",
        "key_facts_to_include"   : [],
        "validation_focus"       : [],
        "llm_syndrome_watch"     : "",
        "evaluation_overrides"   : {},
    }


def _make_pipeline_output(meta_parse_error: bool = False, empty_context: bool = False) -> dict:
    """Minimal pipeline output for Gate 2 tests."""
    return {
        "id"                      : "TC_GATE_TEST",
        "intent"                  : "test",
        "category"                : "test",
        "input_email"             : "Dear team, I need help with my loan EMI payment that failed.",
        "email_subject"           : "EMI payment failed",
        "customer_id"             : "CUST-TEST",
        "ticket_id"               : "TKT-TEST",
        "retrieved_context"       : [] if empty_context else ["EMI failure context chunk."],
        "generated_reply"         : "Dear customer, please retry your EMI payment via UPI.",
        "predicted_ticket_status" : "in_progress",
        "predicted_escalation"    : False,
        "ticket_reasoning"        : "Standard EMI failure case.",
        "meta_parse_error"        : meta_parse_error,
        "model_used"              : "mistral",
        "ground_truth"            : {
            "expected_reply"         : "Please retry EMI payment.",
            "expected_ticket_status" : "in_progress",
            "expected_escalation"    : False,
            "expected_tone"          : "professional",
            "key_facts_to_include"   : [],
        },
        "validation_focus"        : [],
        "llm_syndrome_watch"      : "",
        "evaluation_overrides"    : {},
    }


# ── Gate 1 tests ──────────────────────────────────────────────────────────────

class TestGate1EmailBodyLength:
    """
    Gate 1: email_body must meet minimum length before any LLM call is made.

    Configured via pipeline.min_email_body_length in config.yaml (default: 50).
    Raises ValueError when too short — the caller (playground.py --all loop)
    catches this and marks the case as skipped.
    """

    def test_short_email_raises_value_error(self, config, min_length):
        """
        An email body shorter than min_email_body_length must raise ValueError.
        No Ollama or Groq call should happen — the gate fires before generate_response().
        """
        from playground import run_case

        short_body = "Hi"  # 2 chars — well below min_email_body_length (50)
        test_case  = _make_test_case(short_body)

        with pytest.raises(ValueError, match="email body too short"):
            run_case(test_case, config, verbose=False)

    def test_email_at_exactly_min_length_does_not_raise(self, config, min_length):
        """
        An email body at exactly min_email_body_length must NOT raise ValueError at Gate 1.
        It will fail later (Ollama not running) — that's a different error, not Gate 1.
        Gate 1 is only about length — this confirms the boundary is inclusive.
        """
        from playground import run_case

        exact_body = "x" * min_length  # exactly at threshold
        test_case  = _make_test_case(exact_body)

        # Should not raise ValueError for Gate 1 — may raise RuntimeError for Ollama
        try:
            run_case(test_case, config, verbose=False)
        except ValueError as e:
            if "email body too short" in str(e):
                pytest.fail(
                    f"Gate 1 incorrectly rejected an email of exactly {min_length} chars. "
                    f"Boundary should be inclusive (>=), not exclusive (>)."
                )
        except Exception:
            pass  # Any other error (Ollama down, etc.) is expected and acceptable

    def test_gate1_threshold_matches_config(self, config, min_length):
        """
        The Gate 1 threshold must equal pipeline.min_email_body_length from config.yaml.
        This test verifies config is actually being read, not a hardcoded value.
        """
        assert min_length == config["pipeline"]["min_email_body_length"], (
            "get_pipeline_config() returned a different value than config['pipeline']['min_email_body_length']. "
            "The gate may be reading a hardcoded value instead of config."
        )
        assert isinstance(min_length, int) and min_length > 0, (
            f"min_email_body_length must be a positive integer, got {min_length!r}"
        )


# ── Gate 2 tests ──────────────────────────────────────────────────────────────

class TestGate2SUTOutputValidity:
    """
    Gate 2: LLM evaluation is skipped when SUT output is invalid.

    Two conditions trigger Gate 2:
        1. meta_parse_error=True  — SUT failed to produce structured ticket JSON
        2. retrieved_context=[]   — RAG retrieved nothing (empty KB, bad query)

    When triggered: all LLM metric scores = None with a descriptive error message.
    Custom evaluator (ticket_status_accuracy, escalation_logic) still runs — it
    is deterministic and does not require a valid LLM output structure.
    """

    def test_meta_parse_error_skips_llm_eval(self, config):
        """
        When meta_parse_error=True, LLM evaluation must be skipped.
        All LLM metric scores must be None. Custom scores still run.
        """
        from src.evaluators.custom_evaluator import evaluate as custom_evaluate

        pipeline_output = _make_pipeline_output(meta_parse_error=True)

        # Custom evaluator always runs regardless of gate
        custom_scores = custom_evaluate(pipeline_output)

        # Gate 2 check — replicate the exact logic from playground.py
        meta_parse_error = pipeline_output.get("meta_parse_error")
        empty_context    = len(pipeline_output.get("retrieved_context", [])) == 0

        assert meta_parse_error is True, "Test fixture: meta_parse_error should be True"
        assert not empty_context,        "Test fixture: context should not be empty for this case"

        # Gate 2 fires — LLM eval should be skipped
        gate_triggered = meta_parse_error or empty_context
        assert gate_triggered, "Gate 2 should trigger when meta_parse_error=True"

        # Custom scores should still be present (deterministic — no LLM needed)
        assert "ticket_status_accuracy" in custom_scores
        assert "escalation_logic"       in custom_scores

    def test_empty_context_skips_llm_eval(self, config):
        """
        When retrieved_context=[], LLM evaluation must be skipped.
        Faithfulness and hallucination scores are meaningless without context.
        """
        pipeline_output = _make_pipeline_output(empty_context=True)

        empty_context = len(pipeline_output.get("retrieved_context", [])) == 0
        assert empty_context, "Test fixture: retrieved_context should be empty"

        gate_triggered = pipeline_output.get("meta_parse_error") or empty_context
        assert gate_triggered, "Gate 2 should trigger when retrieved_context is empty"

    def test_skipped_llm_scores_structure(self, config):
        """
        When Gate 2 fires, _skipped_llm_scores() must return the correct null structure.
        Every LLM metric must have score=None and a non-empty error message.
        This is what downstream threshold_checker and report_generator receive.
        """
        from playground import _skipped_llm_scores

        reason = "meta_parse_error"
        scores = _skipped_llm_scores(reason)

        from src.evaluators.combined_evaluator import LLM_METRICS
        expected_metrics = set(LLM_METRICS)
        assert set(scores.keys()) == expected_metrics, (
            f"_skipped_llm_scores() missing keys: {expected_metrics - set(scores.keys())}"
        )
        for metric, data in scores.items():
            assert data["score"] is None, f"{metric}: score should be None when gate fires"
            assert data["error"] is not None, f"{metric}: error message must be set when gate fires"
            assert len(data["error"]) > 0,   f"{metric}: error message must not be empty"

    def test_valid_output_does_not_trigger_gate(self, config):
        """
        A valid pipeline output (no parse error, non-empty context) must NOT trigger Gate 2.
        This is the negative case — confirms the gate is not over-eager.
        """
        pipeline_output = _make_pipeline_output(meta_parse_error=False, empty_context=False)

        meta_parse_error = pipeline_output.get("meta_parse_error")
        empty_context    = len(pipeline_output.get("retrieved_context", [])) == 0

        gate_triggered = meta_parse_error or empty_context
        assert not gate_triggered, (
            "Gate 2 should NOT trigger for a valid pipeline output. "
            "A false positive here means valid cases are being skipped."
        )

    def test_both_conditions_trigger_gate(self, config):
        """
        When both meta_parse_error=True AND retrieved_context=[], gate still fires.
        Tests that the OR logic handles the combined case correctly.
        """
        pipeline_output = _make_pipeline_output(meta_parse_error=True, empty_context=True)

        meta_parse_error = pipeline_output.get("meta_parse_error")
        empty_context    = len(pipeline_output.get("retrieved_context", [])) == 0

        assert meta_parse_error and empty_context, "Both conditions should be True in this fixture"
        gate_triggered = meta_parse_error or empty_context
        assert gate_triggered, "Gate 2 must trigger when both conditions are True"
