"""
data_loader.py
--------------
Loads and joins test data from three separate files:
    data/emails.json       → customer email inputs
    data/context.json      → retrieved context chunks (simulated RAG)
    data/ground_truth.json → expected outputs + validation metadata

Joins all three by test case ID and returns a unified list.
Every other module gets its data through here — nothing reads data files directly.

Usage:
    from src.utils.data_loader import load_test_cases, get_case_by_id

Standalone test:
    python -m src.utils.data_loader
"""

import json
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Fields that must be present and non-empty for a case to be runnable.
# If any are missing the case is skipped before touching the LLM.
REQUIRED_CASE_FIELDS = [
    "email_body",
    "email_subject",
    "retrieved_chunks",
    "expected_ticket_status",
    "expected_escalation",
]


def validate_case_fields(case: dict) -> None:
    """
    Raise ValueError with a clear message if any required field is missing or empty.
    Called after joining all 3 data files — before the case is returned to callers.
    """
    missing = [f for f in REQUIRED_CASE_FIELDS if case.get(f) is None or f not in case]
    if missing:
        raise ValueError(
            f"{case.get('id', '?')}: missing required field(s): {', '.join(missing)} "
            f"— fix in emails.json / context.json / ground_truth.json before running"
        )


def _load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_test_cases(case_ids: Optional[list] = None) -> list:
    """
    Load and join emails, context, and ground_truth by ID.

    Args:
        case_ids: Optional list of IDs to filter (e.g. ["TC001", "TC003"]).
                  If None, returns all cases.

    Returns:
        List of unified test case dicts, each containing:
            id, category, intent, priority
            email_subject, email_body, customer_id, ticket_id
            retrieved_chunks
            expected_reply, expected_ticket_status, expected_escalation
            expected_tone, key_facts_to_include
            validation_focus, llm_syndrome_watch
            evaluation_overrides
    """
    emails_data       = _load_json("emails.json")
    context_data      = _load_json("context.json")
    ground_truth_data = _load_json("ground_truth.json")

    # Index context and ground truth by ID for fast lookup
    context_index = {c["id"]: c for c in context_data["contexts"]}
    gt_index      = {g["id"]: g for g in ground_truth_data["ground_truth"]}

    test_cases = []
    for email in emails_data["emails"]:
        case_id = email["id"]

        if case_ids and case_id not in case_ids:
            continue

        # Validate that matching context and ground truth exist
        if case_id not in context_index:
            print(f"  [WARNING] No context found for {case_id} — skipping")
            continue
        if case_id not in gt_index:
            print(f"  [WARNING] No ground truth found for {case_id} — skipping")
            continue

        ctx = context_index[case_id]
        gt  = gt_index[case_id]

        case = {
            # Identity
            "id"          : case_id,
            "category"    : email["category"],
            "intent"      : email["intent"],
            "priority"    : email["priority"],

            # Input (from emails.json)
            "email_subject" : email["email_subject"],
            "email_body"    : email["email_body"],
            "customer_id"   : email["customer_id"],
            "ticket_id"     : email["ticket_id"],

            # Context (from context.json)
            "retrieved_chunks" : ctx["retrieved_chunks"],

            # Ground truth (from ground_truth.json)
            "expected_reply"         : gt["expected_reply"],
            "expected_ticket_status" : gt["expected_ticket_status"],
            "expected_escalation"    : gt["expected_escalation"],
            "expected_tone"          : gt["expected_tone"],
            "key_facts_to_include"   : gt["key_facts_to_include"],

            # Validation metadata
            "validation_focus"    : gt.get("validation_focus", []),
            "llm_syndrome_watch"  : gt.get("llm_syndrome_watch", ""),

            # Per-case threshold overrides (empty dict = use global config.yaml values)
            "evaluation_overrides": gt.get("evaluation_overrides", {})
        }

        try:
            validate_case_fields(case)
        except ValueError as e:
            print(f"  [SKIPPED] {e}")
            continue

        test_cases.append(case)

    return test_cases


def get_case_by_id(case_id: str) -> dict:
    """
    Load a single test case by ID.

    Args:
        case_id: e.g. "TC001"

    Returns:
        Single unified test case dict.

    Raises:
        ValueError if case_id not found.
    """
    cases = load_test_cases(case_ids=[case_id])
    if not cases:
        raise ValueError(f"Test case '{case_id}' not found in data files.")
    return cases[0]


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint

    console = Console()
    console.print("\n[bold cyan]╔══ DATA LOADER — VERIFICATION ══╗[/bold cyan]\n")

    cases = load_test_cases()
    console.print(f"[bold]Total cases loaded:[/bold] {len(cases)}\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID",        width=6)
    table.add_column("Category",  width=14)
    table.add_column("Intent",    width=28)
    table.add_column("Status",    width=12)
    table.add_column("Escalate",  width=9)
    table.add_column("Overrides", width=10)

    for c in cases:
        has_overrides = "YES" if c["evaluation_overrides"] else "no"
        table.add_row(
            c["id"],
            c["category"],
            c["intent"],
            c["expected_ticket_status"],
            str(c["expected_escalation"]),
            has_overrides
        )

    console.print(table)

    # Show one joined case in detail
    console.print(f"\n[bold yellow]── Sample: TC003 (grievance) ──[/bold yellow]")
    tc003 = get_case_by_id("TC003")
    rprint(f"  Email subject  : {tc003['email_subject']}")
    rprint(f"  Context chunks : {len(tc003['retrieved_chunks'])} chunks")
    rprint(f"  Expected status: {tc003['expected_ticket_status']}")
    rprint(f"  Escalate       : {tc003['expected_escalation']}")
    rprint(f"  Syndrome watch : {tc003['llm_syndrome_watch']}")
    rprint(f"  Overrides      : {tc003['evaluation_overrides']}")

    console.print("\n[bold green]✓ Data loader working — all 3 files joined correctly[/bold green]\n")
