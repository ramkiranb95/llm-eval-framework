"""
conftest.py
-----------
Session-scoped fixtures shared across the entire test suite.

Design principles:
  - Each BSFI test case runs through the full pipeline ONCE per session.
    All test functions read from the cached results dict — no re-execution.
  - Config is wrapped in _Config to suppress pytest's full dict dump in
    failure output. Failures show "<Config>" instead of the entire YAML.
  - TC013 requires a separate simulated-RAG result because the Gate 2 test
    must see an empty retrieved_context. Live ChromaDB returns results for
    that query, defeating the purpose of the test.

Fixtures available to all test files without import:
    config              — full config dict (session-scoped, repr-suppressed)
    pipeline_results    — dict[case_id -> run_case() result] for all 16 TCs
    tc013_simulated     — run_case() result for TC013 with RAG mode forced to simulated
    all_test_cases      — raw list of joined test case dicts
    mock_pipeline_output     — static mock for unit tests (no SUT needed)
    mock_oos_pipeline_output — static mock for out-of-scope unit tests
"""

import copy
import pytest

from src.utils.config_loader import load_config
from src.utils.data_loader import load_test_cases, get_case_by_id
from playground import run_case


# ── Config wrapper — suppresses full YAML dump in pytest failure output ────────

class _Config(dict):
    """Thin dict subclass. __repr__ kept short so pytest failure output is readable."""
    def __repr__(self) -> str:
        return "<Config>"


# ── Session fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config() -> _Config:
    """Full pipeline config, loaded once per session."""
    return _Config(load_config())


@pytest.fixture(scope="session")
def all_test_cases() -> list:
    """All 16 BSFI test cases joined from emails + context + ground_truth."""
    return load_test_cases()


@pytest.fixture(scope="session")
def pipeline_results(config: _Config, all_test_cases: list) -> dict:
    """
    Run all 16 test cases through the full pipeline once and cache results.

    Returns:
        dict[case_id: str -> result: dict]
            result keys: pipeline_output, all_scores, threshold_result

    Cases that raise ValueError (e.g. non-English Gate 1 skip) are stored as:
        {"error": "<reason>", "pipeline_output": None, ...}
    so test functions can assert on expected skips without crashing the fixture.
    """
    import time
    from src.utils.config_loader import get_pipeline_config

    delay   = get_pipeline_config(config).get("inter_case_delay_seconds", 0)
    results = {}

    for test_case in all_test_cases:
        case_id = test_case["id"]
        try:
            results[case_id] = run_case(test_case, config, verbose=False)
        except (ValueError, RuntimeError) as exc:
            # ValueError  — Gate 1 skip (non-English, too short)
            # RuntimeError — SUT/Judge rate limit exhausted after retries
            results[case_id] = {
                "error"            : str(exc),
                "pipeline_output"  : None,
                "all_scores"       : {},
                "threshold_result" : {},
            }

        if delay > 0:
            time.sleep(delay)

    return results


@pytest.fixture(scope="session")
def tc013_simulated(config: _Config) -> dict:
    """
    TC013 result with RAG mode forced to 'simulated'.

    Gate 2 (empty context) only fires when retrieved_chunks from context.json
    are used directly — live ChromaDB returns matches for this query.
    This fixture isolates that behaviour without affecting other TC runs.
    """
    try:
        test_case = get_case_by_id("TC013")
    except ValueError:
        pytest.skip("TC013 not found in data files — skipping Gate 2 fixture")
    simulated_config  = _Config(copy.deepcopy(dict(config)))
    simulated_config["rag"]["mode"] = "simulated"
    return run_case(test_case, simulated_config, verbose=False)


# ── Function-scoped fixtures — fresh per unit test ─────────────────────────────

@pytest.fixture
def mock_pipeline_output() -> dict:
    """
    Static pipeline output for evaluator unit tests that do not need a live SUT.
    Mirrors the structure returned by crm_responder.generate_response().
    """
    return {
        "id"                      : "TC001",
        "intent"                  : "loan_eligibility_query",
        "category"                : "loan_query",
        "input_email"             : (
            "Dear Team, I would like to know if I am eligible for a personal loan "
            "of Rs. 2,00,000. My monthly income is Rs. 25,000. Please advise."
        ),
        "email_subject"           : "Enquiry regarding personal loan eligibility",
        "customer_id"             : "CUST-10234",
        "ticket_id"               : "TKT-88901",
        "retrieved_context"       : [
            "Loan eligibility requires minimum 3 years of continuous employment.",
            "Maximum personal loan amount is Rs. 5,00,000 for salaried applicants.",
            "Required documents: ID proof, salary slips (last 3 months), bank statements.",
        ],
        "generated_reply"         : (
            "Dear Ramesh, thank you for reaching out. Based on your details, "
            "you are eligible for a personal loan. Please submit your ID proof, "
            "last 3 months salary slips, and bank statements to proceed. "
            "Regards, Customer Support Team"
        ),
        "predicted_ticket_status" : "in_progress",
        "predicted_escalation"    : False,
        "ticket_reasoning"        : "Standard loan eligibility query, processing in progress.",
        "meta_parse_error"        : False,
        "model_used"              : "mistral",
        "ground_truth"            : {
            "expected_reply"         : "Confirm eligibility and list required documents.",
            "expected_ticket_status" : "in_progress",
            "expected_escalation"    : False,
            "expected_tone"          : "professional",
            "key_facts_to_include"   : [
                "loan eligibility confirmed",
                "documents required",
                "salary slips",
            ],
        },
        "validation_focus"     : [],
        "llm_syndrome_watch"   : "",
        "evaluation_overrides" : {},
    }


@pytest.fixture
def mock_oos_pipeline_output() -> dict:
    """Static pipeline output for out-of-scope unit tests (TC010-style)."""
    return {
        "id"                      : "TC010",
        "intent"                  : "out_of_scope_query",
        "category"                : "loan_query",
        "input_email"             : (
            "Hi, I wanted to ask about which stocks I should buy this week. "
            "I have Rs. 50,000 to invest. Please advise."
        ),
        "email_subject"           : "Investment advice needed",
        "customer_id"             : "CUST-99999",
        "ticket_id"               : "TKT-00001",
        "retrieved_context"       : [
            "We do not provide investment advisory services.",
            "For investment advice, please consult a SEBI-registered advisor.",
        ],
        "generated_reply"         : (
            "Dear Customer, we appreciate your query. However, providing investment "
            "advice is outside our scope. Please consult a SEBI-registered investment "
            "advisor for guidance. Regards, Customer Support Team"
        ),
        "predicted_ticket_status" : "resolved",
        "predicted_escalation"    : False,
        "ticket_reasoning"        : "Out-of-scope query, redirected to SEBI advisor.",
        "meta_parse_error"        : False,
        "model_used"              : "mistral",
        "ground_truth"            : {
            "expected_reply"         : "Redirect to SEBI advisor.",
            "expected_ticket_status" : "resolved",
            "expected_escalation"    : False,
            "expected_tone"          : "professional",
            "key_facts_to_include"   : [
                "out of scope acknowledged",
                "SEBI advisor recommendation",
            ],
        },
        "validation_focus"     : ["out_of_scope_handling"],
        "llm_syndrome_watch"   : "confabulation",
        "evaluation_overrides" : {},
    }
