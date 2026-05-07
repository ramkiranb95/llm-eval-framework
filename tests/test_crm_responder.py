"""
test_crm_responder.py
---------------------
Tests for src/pipeline/crm_responder.py.

Split into two sections:

    Unit tests   — test _parse_combined_output() in isolation.
                   No Ollama needed. Fast. Always runnable.

    Integration tests — test generate_response() end to end.
                   REQUIRES Ollama to be running at http://localhost:11434
                   with the configured SUT model pulled.

                   If Ollama is not running these tests FAIL with a clear
                   RuntimeError message rather than being silently skipped.
                   This is intentional — a skipped test looks like a passing
                   test. Failing loudly tells you exactly what is missing.

                   To run Ollama before executing:
                       ollama serve
                       ollama pull mistral

Run all:
    pytest tests/test_crm_responder.py -v

Run unit tests only (no Ollama):
    pytest tests/test_crm_responder.py -v -k "not integration"

Run integration tests only:
    pytest tests/test_crm_responder.py -v -m integration
"""

import pytest
from src.pipeline.crm_responder import _parse_combined_output, generate_response


# ── Unit tests — _parse_combined_output ──────────────────────────────────────

class TestParseCombinedOutput:
    """
    Tests for the LLM output parser.
    These are the most important unit tests in this file — the parser is the
    fragile layer between raw LLM text and structured data.
    """

    def test_well_formed_output_parses_correctly(self):
        raw = """[REPLY]
Dear Ramesh, your loan application is being processed.
Regards, Customer Support Team
[/REPLY]
[META]
{"ticket_status": "in_progress", "escalation": false, "reasoning": "Standard loan query."}
[/META]"""
        reply, ticket = _parse_combined_output(raw)
        assert "Ramesh" in reply
        assert "Customer Support Team" in reply
        assert ticket["ticket_status"] == "in_progress"
        assert ticket["escalation"]    is False
        assert ticket["parse_error"]   is False

    def test_missing_closing_tags_still_parses(self):
        # mistral often omits [/REPLY] and [/META]
        raw = """[REPLY]
Dear Customer, your EMI has been noted.
Regards, Customer Support Team
[META]
{"ticket_status": "in_progress", "escalation": false, "reasoning": "EMI query."}"""
        reply, ticket = _parse_combined_output(raw)
        assert "Customer Support Team" in reply
        assert ticket["ticket_status"] == "in_progress"

    def test_escalated_case_parses_correctly(self):
        raw = """[REPLY]
Dear Priya, we take your complaint seriously and are escalating this.
Regards, Customer Support Team
[/REPLY]
[META]
{"ticket_status": "escalated", "escalation": true, "reasoning": "Interest rate dispute requires escalation."}
[/META]"""
        reply, ticket = _parse_combined_output(raw)
        assert ticket["ticket_status"] == "escalated"
        assert ticket["escalation"]    is True
        assert ticket["parse_error"]   is False

    def test_markdown_code_fence_stripped_from_meta(self):
        raw = """[REPLY]
Your query has been received.
Regards, Customer Support Team
[/REPLY]
[META]
```json
{"ticket_status": "resolved", "escalation": false, "reasoning": "Simple query resolved."}
```
[/META]"""
        reply, ticket = _parse_combined_output(raw)
        assert ticket["ticket_status"] == "resolved"
        assert ticket["parse_error"]   is False

    def test_missing_meta_block_returns_default_with_parse_error(self):
        raw = """[REPLY]
Dear Customer, your query has been noted.
Regards, Customer Support Team
[/REPLY]"""
        reply, ticket = _parse_combined_output(raw)
        assert "Customer Support Team" in reply
        assert ticket["parse_error"]           is True
        assert ticket["ticket_status"]         == "in_progress"
        assert ticket["escalation"]            is False

    def test_corrupt_json_in_meta_returns_default_with_parse_error(self):
        raw = """[REPLY]
Your request is being processed.
Regards, Customer Support Team
[/REPLY]
[META]
{ticket_status: broken json here
[/META]"""
        reply, ticket = _parse_combined_output(raw)
        assert ticket["parse_error"] is True

    def test_reply_text_is_trimmed(self):
        raw = """[REPLY]
   Dear Ramesh, here is your information.
   Regards, Customer Support Team
   [/REPLY]
[META]
{"ticket_status": "resolved", "escalation": false, "reasoning": "Done."}
[/META]"""
        reply, ticket = _parse_combined_output(raw)
        assert not reply.startswith(" ")
        assert not reply.endswith(" ")

    def test_out_of_scope_resolved_status_parses(self):
        raw = """[REPLY]
Dear Customer, investment advice is outside our scope.
Please consult a SEBI-registered advisor.
Regards, Customer Support Team
[/REPLY]
[META]
{"ticket_status": "resolved", "escalation": false, "reasoning": "Out-of-scope query redirected."}
[/META]"""
        reply, ticket = _parse_combined_output(raw)
        assert ticket["ticket_status"] == "resolved"
        assert ticket["escalation"]    is False


# ── Integration tests — generate_response ────────────────────────────────────
# These tests require Ollama to be running.
# They FAIL with a RuntimeError if Ollama is not available — by design.

@pytest.mark.integration
class TestGenerateResponseIntegration:

    def test_returns_all_required_keys(self, config, all_test_cases):
        test_case = next(c for c in all_test_cases if c["id"] == "TC001")
        result    = generate_response(test_case, config)

        required_keys = [
            "id", "category", "intent", "input_email", "email_subject",
            "customer_id", "ticket_id", "retrieved_context",
            "generated_reply", "predicted_ticket_status", "predicted_escalation",
            "ticket_reasoning", "meta_parse_error", "model_used",
            "ground_truth", "validation_focus", "llm_syndrome_watch",
            "evaluation_overrides",
        ]
        for key in required_keys:
            assert key in result, f"Missing key in pipeline output: '{key}'"

    def test_generated_reply_is_non_empty_string(self, config, all_test_cases):
        test_case = next(c for c in all_test_cases if c["id"] == "TC001")
        result    = generate_response(test_case, config)
        assert isinstance(result["generated_reply"], str)
        assert len(result["generated_reply"]) > 0

    def test_predicted_ticket_status_is_valid(self, config, all_test_cases):
        test_case      = next(c for c in all_test_cases if c["id"] == "TC001")
        result         = generate_response(test_case, config)
        valid_statuses = {"open", "in_progress", "escalated", "resolved"}
        assert result["predicted_ticket_status"] in valid_statuses, (
            f"Invalid ticket_status: '{result['predicted_ticket_status']}'"
        )

    def test_predicted_escalation_is_boolean(self, config, all_test_cases):
        test_case = next(c for c in all_test_cases if c["id"] == "TC001")
        result    = generate_response(test_case, config)
        assert isinstance(result["predicted_escalation"], bool)

    def test_meta_parse_error_is_boolean(self, config, all_test_cases):
        test_case = next(c for c in all_test_cases if c["id"] == "TC001")
        result    = generate_response(test_case, config)
        assert isinstance(result["meta_parse_error"], bool)

    def test_ground_truth_passthrough_is_intact(self, config, all_test_cases):
        test_case = next(c for c in all_test_cases if c["id"] == "TC001")
        result    = generate_response(test_case, config)
        gt        = result["ground_truth"]
        assert "expected_ticket_status" in gt
        assert "expected_escalation"    in gt
        assert "key_facts_to_include"   in gt
        assert gt["expected_ticket_status"] == test_case["expected_ticket_status"]
        assert gt["expected_escalation"]    == test_case["expected_escalation"]

    def test_escalation_case_correctly_flagged(self, config, all_test_cases):
        # TC003 is an interest rate dispute — expected to be escalated
        test_case = next(c for c in all_test_cases if c["id"] == "TC003")
        result    = generate_response(test_case, config)
        assert result["predicted_ticket_status"] in {"escalated", "in_progress"}, (
            f"TC003 (grievance) should be escalated or in_progress, "
            f"got '{result['predicted_ticket_status']}'"
        )

    @pytest.mark.parametrize("case_id", ["TC001", "TC002", "TC003", "TC010"])
    def test_pipeline_runs_for_key_cases(self, config, all_test_cases, case_id):
        test_case = next(c for c in all_test_cases if c["id"] == case_id)
        result    = generate_response(test_case, config)
        assert result["id"]              == case_id
        assert result["generated_reply"] != ""
        assert result["predicted_ticket_status"] in {"open", "in_progress", "escalated", "resolved"}
