"""
conftest.py
-----------
Shared pytest fixtures for the LLM Eval Framework test suite.

Fixtures defined here are automatically available to all test files
without any import — pytest discovers conftest.py by convention.

Session-scoped fixtures (config, all_test_cases) are initialised once
per pytest run, not once per test. This avoids re-reading YAML and JSON
files on every test case.

Usage:
    Any test function that declares a fixture name as a parameter
    receives it automatically.

    def test_something(config, mock_pipeline_output):
        ...
"""

import pytest
from src.utils.config_loader import load_config
from src.utils.data_loader import load_test_cases, get_case_by_id


# ── Session-scoped fixtures — initialised once per run ───────────────────────

@pytest.fixture(scope="session")
def config():
    """Full config dict loaded from config/config.yaml."""
    return load_config()


@pytest.fixture(scope="session")
def all_test_cases():
    """All 10 BSFI test cases joined from emails + context + ground_truth."""
    return load_test_cases()


# ── Function-scoped fixtures — fresh per test ────────────────────────────────

@pytest.fixture
def mock_pipeline_output():
    """
    A valid pipeline output dict for evaluator tests that do not need Ollama.

    Mirrors the structure returned by crm_responder.generate_response().
    Use this whenever you want to test an evaluator in isolation.
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
def mock_oos_pipeline_output():
    """
    Pipeline output for an out-of-scope query (TC010-style).
    Used to test out_of_scope_handling metric.
    """
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
