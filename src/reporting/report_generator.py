"""
report_generator.py
-------------------
Generates two outputs from a completed evaluation run:

1. Terminal — rich table showing per-case, per-metric pass/fail with colours,
   followed by the release gate banner.

2. JSON file — full per-case traceability saved to:
   reports/run_<YYYYMMDD_HHMMSS>.json

The JSON structure is:
{
    "run_id"        : "run_20260416_143022",
    "timestamp"     : "2026-04-16T14:30:22",
    "total_cases"   : 10,
    "gate"          : { ...release gate result... },
    "cases"         : [
        {
            "id"       : "TC001",
            "category" : "loan_query",
            "intent"   : "loan_eligibility_query",
            "metrics"  : { ...threshold_checker output... },
            "generated_reply"         : "...",
            "predicted_ticket_status" : "in_progress",
            "predicted_escalation"    : false,
            "ground_truth": { ... }
        },
        ...
    ]
}

Usage:
    from src.reporting.report_generator import generate_report

Standalone test:
    python -m src.reporting.report_generator
"""

import json
from datetime import datetime
from pathlib import Path


REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def _metric_colour(metric_name: str, data: dict) -> str:
    """Return a Rich colour tag based on pass/fail/critical."""
    passed   = data.get("passed")
    critical = data.get("critical", False)
    score    = data.get("score")

    if score is None:
        return "yellow"       # judge LLM failed — amber warning
    if passed is True:
        return "green"
    if passed is False and critical:
        return "red"          # critical failure — bright red
    if passed is False:
        return "yellow"       # non-critical failure — amber
    return "dim"


def print_terminal_report(
    case_results          : list,
    case_threshold_results: list,
    gate_result           : dict
) -> None:
    """
    Print a rich terminal table of evaluation results.

    Args:
        case_results           : list of pipeline_output dicts (from crm_responder)
        case_threshold_results : list of threshold_checker outputs (one per case)
        gate_result            : dict from release_gate.evaluate_gate()
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint

    console = Console()

    # ── Collect all metric names across all cases ─────────────────────────────
    all_metrics = []
    for cr in case_threshold_results:
        for k in cr:
            if k != "id" and k not in all_metrics:
                all_metrics.append(k)

    # ── Build results table ─────────────────────���─────────────────────────────
    table = Table(
        title="Evaluation Results",
        show_header=True,
        header_style="bold magenta",
        show_lines=True
    )
    table.add_column("Case",     width=7,  style="bold")
    table.add_column("Category", width=13)
    table.add_column("Intent",   width=28)

    for m in all_metrics:
        table.add_column(m[:18], width=9)

    for i, cr in enumerate(case_threshold_results):
        case_id  = cr.get("id", "—")
        # Find matching pipeline output for category/intent
        po = next((r for r in case_results if r.get("id") == case_id), {})

        row = [
            case_id,
            po.get("category", "—"),
            po.get("intent", "—")[:26]
        ]

        for m in all_metrics:
            data  = cr.get(m, {})
            score = data.get("score")
            colour = _metric_colour(m, data)

            if score is None:
                cell = f"[{colour}]—[/{colour}]"
            elif m == "hallucination":
                cell = f"[{colour}]{score:.2f}↓[/{colour}]"
            else:
                cell = f"[{colour}]{score:.2f}[/{colour}]"

            row.append(cell)

        table.add_row(*row)

    console.print(table)

    # ── Gate banner ──────────────────────────────────��────────────────────��───
    gate_colour = "green" if gate_result["passed"] else "red"
    console.print(Panel(
        f"[{gate_colour}]{gate_result['message']}[/{gate_colour}]\n"
        f"Cases passed: {gate_result['cases_passed']} / {gate_result['total_cases']}",
        border_style=gate_colour,
        title="Release Gate"
    ))

    # Surface META parse errors — these cause ticket_status/escalation to default silently
    parse_errors = [r for r in case_results if r.get("meta_parse_error")]
    if parse_errors:
        console.print("\n[yellow bold]META parse warnings (ticket status defaulted to in_progress):[/yellow bold]")
        for r in parse_errors:
            rprint(f"  [yellow]⚠[/yellow] {r['id']} — {r.get('ticket_reasoning', '')}")

    if not gate_result["passed"]:
        console.print("\n[red bold]Critical failures:[/red bold]")
        for f in gate_result["critical_failures"]:
            rprint(
                f"  [red]✗[/red] {f['case_id']} / {f['metric']}"
                f" — score={f['score']} threshold={f['threshold']}"
                + (f" [{f['error']}]" if f.get("error") else "")
            )


def save_json_report(
    case_results          : list,
    case_threshold_results: list,
    gate_result           : dict
) -> Path:
    """
    Save full evaluation results to a timestamped JSON file in reports/.

    Returns:
        Path to the saved report file.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id     = f"run_{timestamp}"
    report_dir = _ensure_reports_dir()
    report_path = report_dir / f"{run_id}.json"

    # Build per-case traceability records
    cases_data = []
    for i, threshold_result in enumerate(case_threshold_results):
        case_id = threshold_result.get("id", f"case_{i}")
        po = next((r for r in case_results if r.get("id") == case_id), {})

        cases_data.append({
            "id"                      : case_id,
            "category"                : po.get("category"),
            "intent"                  : po.get("intent"),
            "generated_reply"         : po.get("generated_reply"),
            "predicted_ticket_status" : po.get("predicted_ticket_status"),
            "predicted_escalation"    : po.get("predicted_escalation"),
            "ticket_reasoning"        : po.get("ticket_reasoning"),
            "meta_parse_error"        : po.get("meta_parse_error", False),
            "model_used"              : po.get("model_used"),
            "ground_truth"            : po.get("ground_truth", {}),
            "validation_focus"        : po.get("validation_focus", []),
            "llm_syndrome_watch"      : po.get("llm_syndrome_watch", ""),
            "evaluation_overrides"    : po.get("evaluation_overrides", {}),
            "metrics"                 : {
                k: v for k, v in threshold_result.items() if k != "id"
            }
        })

    report = {
        "run_id"      : run_id,
        "timestamp"   : datetime.now().isoformat(),
        "total_cases" : len(case_results),
        "gate"        : gate_result,
        "cases"       : cases_data
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report_path


def generate_report(
    case_results          : list,
    case_threshold_results: list,
    gate_result           : dict,
    save_json             : bool = True
) -> None:
    """
    Print terminal report and optionally save JSON.

    Args:
        case_results           : pipeline output dicts (from crm_responder)
        case_threshold_results : threshold checker dicts (one per case, with "id")
        gate_result            : from release_gate.evaluate_gate()
        save_json              : if True, writes reports/run_<timestamp>.json
    """
    from rich import print as rprint

    print_terminal_report(case_results, case_threshold_results, gate_result)

    if save_json:
        path = save_json_report(case_results, case_threshold_results, gate_result)
        rprint(f"\n[dim]Report saved: {path}[/dim]")


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from src.utils.config_loader import load_config

    console = Console()
    console.print("\n[bold cyan]╔══ REPORT GENERATOR — VERIFICATION ══╗[/bold cyan]\n")

    config = load_config()

    # Mock minimal pipeline outputs and threshold results
    mock_case_results = [
        {
            "id": "TC001", "category": "loan_query", "intent": "loan_eligibility_query",
            "generated_reply": "Dear Ramesh, you are eligible for a loan. Documents: salary slips, bank statements. Regards, CX Team",
            "predicted_ticket_status": "in_progress", "predicted_escalation": False,
            "ticket_reasoning": "Standard query", "model_used": "llama3.2:3b",
            "ground_truth": {"expected_ticket_status": "in_progress", "expected_escalation": False,
                             "expected_reply": "...", "expected_tone": "helpful", "key_facts_to_include": []},
            "validation_focus": ["Cat-1: coherence"], "llm_syndrome_watch": "confabulation",
            "evaluation_overrides": {}
        },
        {
            "id": "TC008", "category": "grievance", "intent": "mis_selling_complaint",
            "generated_reply": "Dear Ravi, we apologise. Grievance ref: GRV-88908. GRO will contact you. Regards, CX Team",
            "predicted_ticket_status": "escalated", "predicted_escalation": True,
            "ticket_reasoning": "Escalated to GRO", "model_used": "llama3.2:3b",
            "ground_truth": {"expected_ticket_status": "escalated", "expected_escalation": True,
                             "expected_reply": "...", "expected_tone": "empathetic_formal", "key_facts_to_include": []},
            "validation_focus": ["Cat-2: faithfulness"], "llm_syndrome_watch": "confabulation",
            "evaluation_overrides": {"faithfulness_threshold": 0.90}
        }
    ]

    mock_threshold_results = [
        {
            "id"                      : "TC001",
            "faithfulness"            : {"score": 0.88, "passed": True,  "critical": True,  "threshold": 0.85, "inverted": False, "error": None},
            "hallucination"           : {"score": 0.07, "passed": True,  "critical": True,  "threshold": 0.10, "inverted": True,  "error": None},
            "ticket_status_accuracy"  : {"score": 1.00, "passed": True,  "critical": True,  "threshold": 1.00, "inverted": False, "error": None},
            "escalation_logic"        : {"score": 1.00, "passed": True,  "critical": True,  "threshold": 1.00, "inverted": False, "error": None},
            "key_facts_coverage"      : {"score": 0.50, "passed": False, "critical": False, "threshold": 0.75, "inverted": False, "error": None},
        },
        {
            "id"                      : "TC008",
            "faithfulness"            : {"score": 0.85, "passed": False, "critical": True,  "threshold": 0.90, "inverted": False, "error": None},  # override threshold
            "hallucination"           : {"score": 0.03, "passed": True,  "critical": True,  "threshold": 0.05, "inverted": True,  "error": None},
            "ticket_status_accuracy"  : {"score": 1.00, "passed": True,  "critical": True,  "threshold": 1.00, "inverted": False, "error": None},
            "escalation_logic"        : {"score": 1.00, "passed": True,  "critical": True,  "threshold": 1.00, "inverted": False, "error": None},
        }
    ]

    from src.scoring.release_gate import evaluate_gate
    gate = evaluate_gate(mock_threshold_results, config)

    generate_report(mock_case_results, mock_threshold_results, gate, save_json=True)

    console.print("\n[bold green]✓ Report generator working[/bold green]\n")
