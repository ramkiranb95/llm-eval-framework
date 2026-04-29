"""
release_gate.py
---------------
Go / No-Go decision logic for the full evaluation run.

Policy: all_critical_must_pass
    If ANY critical metric fails across ANY test case → gate FAILS.
    All non-critical metrics can fail without blocking the gate.

The gate operates on the aggregated results of all test cases.
Individual case results are summarised into a single verdict.

Usage:
    from src.scoring.release_gate import evaluate_gate

Standalone test:
    python -m src.scoring.release_gate
"""


def evaluate_gate(case_threshold_results: list, config: dict) -> dict:
    """
    Evaluate the release gate across all test cases.

    Args:
        case_threshold_results : list of per-case threshold check results.
            Each element is the dict returned by threshold_checker.check_thresholds()
            for one test case, augmented with the case "id".
            Example:
            [
                {
                    "id": "TC001",
                    "faithfulness":     {"score": 0.87, "passed": True,  "critical": True, ...},
                    "hallucination":    {"score": 0.04, "passed": True,  "critical": True, ...},
                    "ticket_status..":  {"score": 1.00, "passed": True,  "critical": True, ...},
                    ...
                },
                ...
            ]
        config : full config dict from load_config() (reserved for future policy options)

    Returns:
        {
            "passed"           : bool   — True = gate passes, False = blocked
            "message"          : str    — one-line summary
            "critical_failures": list   — [{case_id, metric, score, threshold}, ...]
            "total_cases"      : int
            "cases_passed"     : int    — all critical metrics passed for this case
            "cases_failed"     : int
        }
    """
    critical_failures = []
    cases_passed      = 0
    cases_failed      = 0

    for case_result in case_threshold_results:
        case_id       = case_result.get("id", "unknown")
        case_failed   = False

        for metric_name, data in case_result.items():
            if metric_name == "id":
                continue
            if not isinstance(data, dict):
                continue

            critical = data.get("critical", False)
            passed   = data.get("passed", None)

            if critical and passed is False:
                critical_failures.append({
                    "case_id"   : case_id,
                    "metric"    : metric_name,
                    "score"     : data.get("score"),
                    "threshold" : data.get("threshold"),
                    "error"     : data.get("error")
                })
                case_failed = True

        if case_failed:
            cases_failed += 1
        else:
            cases_passed += 1

    gate_passed = len(critical_failures) == 0
    total_cases = len(case_threshold_results)

    if gate_passed:
        message = f"✅ RELEASE GATE PASSED — all {total_cases} cases cleared all critical metrics"
    else:
        message = (
            f"❌ RELEASE GATE FAILED — {len(critical_failures)} critical failure(s) "
            f"across {cases_failed} of {total_cases} case(s)"
        )

    return {
        "passed"           : gate_passed,
        "message"          : message,
        "critical_failures": critical_failures,
        "total_cases"      : total_cases,
        "cases_passed"     : cases_passed,
        "cases_failed"     : cases_failed
    }


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    from src.utils.config_loader import load_config

    console = Console()
    console.print("\n[bold cyan]╔══ RELEASE GATE — VERIFICATION ══╗[/bold cyan]\n")

    config = load_config()

    # ── Scenario 1: all pass ──────────────────────────────────────────────────
    console.print("[bold yellow]── Scenario 1: All critical metrics pass ──[/bold yellow]")
    all_pass = [
        {
            "id"                      : "TC001",
            "faithfulness"            : {"score": 0.92, "passed": True,  "critical": True,  "threshold": 0.85},
            "hallucination"           : {"score": 0.05, "passed": True,  "critical": True,  "threshold": 0.10},
            "ticket_status_accuracy"  : {"score": 1.00, "passed": True,  "critical": True,  "threshold": 1.00},
            "escalation_logic"        : {"score": 1.00, "passed": True,  "critical": True,  "threshold": 1.00},
            "key_facts_coverage"      : {"score": 0.65, "passed": False, "critical": False, "threshold": 0.75},
        },
        {
            "id"                      : "TC003",
            "faithfulness"            : {"score": 0.91, "passed": True,  "critical": True,  "threshold": 0.90},
            "hallucination"           : {"score": 0.03, "passed": True,  "critical": True,  "threshold": 0.05},
            "ticket_status_accuracy"  : {"score": 1.00, "passed": True,  "critical": True,  "threshold": 1.00},
            "escalation_logic"        : {"score": 1.00, "passed": True,  "critical": True,  "threshold": 1.00},
        }
    ]
    result1 = evaluate_gate(all_pass, config)
    colour1 = "green" if result1["passed"] else "red"
    console.print(f"[{colour1}]{result1['message']}[/{colour1}]")
    rprint(f"  Cases passed: {result1['cases_passed']} / {result1['total_cases']}")

    # ── Scenario 2: one critical failure ─────────────────────────────────────
    console.print(f"\n[bold yellow]── Scenario 2: TC008 escalation_logic fails (critical) ──[/bold yellow]")
    with_failure = [
        {
            "id"                      : "TC008",
            "faithfulness"            : {"score": 0.93, "passed": True,  "critical": True,  "threshold": 0.90},
            "hallucination"           : {"score": 0.02, "passed": True,  "critical": True,  "threshold": 0.05},
            "ticket_status_accuracy"  : {"score": 1.00, "passed": True,  "critical": True,  "threshold": 1.00},
            "escalation_logic"        : {"score": 0.00, "passed": False, "critical": True,  "threshold": 1.00},  # ← FAIL
        }
    ]
    result2 = evaluate_gate(with_failure, config)
    colour2 = "green" if result2["passed"] else "red"
    console.print(f"[{colour2}]{result2['message']}[/{colour2}]")
    rprint(f"  Critical failures: {result2['critical_failures']}")

    console.print("\n[bold green]✓ Release gate working[/bold green]\n")
