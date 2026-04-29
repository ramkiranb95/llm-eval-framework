"""
playground.py
-------------
The local BSFI hands-on sandbox. Run any single test case or all 10 instantly.

This is your fast feedback loop — no pytest, no full report file.
Use this to:
  - Try a new test case and see all scores immediately
  - Tune thresholds in config.yaml and see the impact
  - Observe which LLM syndromes appear in which cases
  - Understand what RAGAs / DeepEval are actually measuring

Usage:
    python playground.py TC001          # run single case
    python playground.py TC003 TC008    # run specific cases
    python playground.py --all          # run all 10 cases
    python playground.py --list         # list available cases
"""

import sys
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import print as rprint

from src.utils.config_loader import load_config
from src.utils.data_loader import load_test_cases, get_case_by_id
from src.pipeline.crm_responder import generate_response
from src.evaluators.custom_evaluator import evaluate as custom_evaluate
from src.scoring.threshold_checker import check_thresholds
from src.scoring.release_gate import evaluate_gate


console = Console()


# ── Run a single case ─────────────────────────────────────────────────────────

def run_case(test_case: dict, config: dict, verbose: bool = True) -> dict:
    """
    Run one test case through the full evaluation pipeline.

    Returns:
        {
            "pipeline_output"   : dict from crm_responder
            "all_scores"        : merged scores from all evaluators
            "threshold_result"  : dict from threshold_checker (with "id")
        }
    """
    case_id = test_case["id"]

    if verbose:
        console.print(Rule(f"[bold cyan]{case_id} — {test_case['intent']}[/bold cyan]"))
        console.print(f"[bold]Category:[/bold] {test_case['category']}  |  [bold]Priority:[/bold] {test_case['priority']}")
        console.print(f"[bold]Email:[/bold]    {test_case['email_subject']}")
        console.print(f"[dim]Syndrome watch: {test_case.get('llm_syndrome_watch', '—')}[/dim]\n")

    # ── Step 1: Generate CRM reply ────────────────────────────────────────────
    pipeline_output = generate_response(test_case, config)

    if verbose:
        console.print(Panel(
            pipeline_output["generated_reply"],
            title="[green]Generated Reply[/green]",
            border_style="green"
        ))
        rprint(f"\n[bold]Predicted status:[/bold] [yellow]{pipeline_output['predicted_ticket_status']}[/yellow]  "
               f"[bold]Escalation:[/bold] [yellow]{pipeline_output['predicted_escalation']}[/yellow]")
        rprint(f"[bold]Expected  status:[/bold] [cyan]{pipeline_output['ground_truth']['expected_ticket_status']}[/cyan]  "
               f"[bold]Escalation:[/bold] [cyan]{pipeline_output['ground_truth']['expected_escalation']}[/cyan]")

    # ── Step 2: Evaluate ─────────────────────────────────────────────────────
    eval_mode = config.get("evaluation", {}).get("mode", "separate")

    if verbose:
        mode_label = "combined (1 LLM call)" if eval_mode == "combined" else "separate (RAGAs + DeepEval)"
        console.print(f"\n[dim]Running evaluators — mode: {mode_label}...[/dim]")

    # Custom evaluator always runs — deterministic, no LLM
    custom_scores = custom_evaluate(pipeline_output)

    # LLM-based evaluators — routed by mode
    if eval_mode == "combined":
        from src.evaluators.combined_evaluator import evaluate as combined_evaluate
        llm_scores = combined_evaluate(pipeline_output, config)
    else:
        raise NotImplementedError(
            f"evaluation.mode '{eval_mode}' is not supported in Tier 1.\n"
            "  Only mode: 'combined' is integrated into the Tier 1 pipeline.\n"
            "  Separate RAGAs and DeepEval evaluators are Tier 2 work-in-progress:\n"
            "    src/evaluators/experimental/ragas_evaluator.py\n"
            "    src/evaluators/experimental/deepeval_evaluator.py\n"
            "  Set evaluation.mode: 'combined' in config/config.yaml to proceed."
        )

    # Merge all scores into one flat dict
    all_scores = {}
    all_scores.update(custom_scores)
    all_scores.update(llm_scores)

    # ── Step 3: Threshold check ──────────────────────────────────────────────
    overrides       = pipeline_output.get("evaluation_overrides", {})
    threshold_result = check_thresholds(all_scores, config, overrides)
    threshold_result["id"] = case_id  # tag with case_id for release gate

    # ── Step 4: Print scores table ───────────────────────────────────────────
    if verbose:
        _print_scores_table(threshold_result, overrides)

    return {
        "pipeline_output"  : pipeline_output,
        "all_scores"       : all_scores,
        "threshold_result" : threshold_result
    }


def _print_scores_table(threshold_result: dict, overrides: dict) -> None:
    """Print per-metric scores with pass/fail colours."""
    table = Table(show_header=True, header_style="bold magenta", show_lines=False)
    table.add_column("Metric",    min_width=24)
    table.add_column("Score",     min_width=7)
    table.add_column("Threshold", min_width=9)
    table.add_column("Pass",      min_width=5)
    table.add_column("Critical",  min_width=8)
    table.add_column("Notes",     min_width=30)

    for metric, data in threshold_result.items():
        if metric == "id":
            continue

        score     = data.get("score")
        threshold = data.get("threshold")
        passed    = data.get("passed")
        critical  = data.get("critical", False)
        inverted  = data.get("inverted", False)
        error     = data.get("error") or data.get("notes", "")

        # Colour
        if score is None:
            colour = "yellow"
        elif passed is True:
            colour = "green"
        elif passed is False and critical:
            colour = "red"
        else:
            colour = "yellow"

        # Override indicator
        override_key = f"{metric}_threshold"
        threshold_str = f"{threshold:.2f}" if threshold is not None else "—"
        if override_key in overrides:
            threshold_str += " [dim]†[/dim]"  # dagger = per-case override

        table.add_row(
            metric,
            f"[{colour}]{score:.3f}[/{colour}]" if score is not None else f"[{colour}]—[/{colour}]",
            threshold_str,
            f"[{colour}]{'✓' if passed else '✗' if passed is False else '?'}[/{colour}]",
            "[red]YES[/red]" if critical else "no",
            (f"↓ inverted  " if inverted else "") + (str(error)[:36] if error else "")
        )

    console.print(table)
    if any(f"{m}_threshold" in overrides for m in threshold_result if m != "id"):
        console.print("[dim]† per-case override (ground_truth.json evaluation_overrides)[/dim]")


# ── List available cases ──────────────────────────────────────────────────────

def list_cases() -> None:
    cases = load_test_cases()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID",       width=7)
    table.add_column("Category", width=14)
    table.add_column("Intent",   width=30)
    table.add_column("Priority", width=10)
    table.add_column("Escalate", width=9)
    table.add_column("Overrides",width=10)
    for c in cases:
        has_overrides = "YES" if c.get("evaluation_overrides") else "no"
        table.add_row(
            c["id"], c["category"], c["intent"], c["priority"],
            str(c["expected_escalation"]), has_overrides
        )
    console.print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Eval Framework — BSFI CRM Playground"
    )
    parser.add_argument(
        "cases",
        nargs="*",
        help="Test case ID(s) to run, e.g. TC001 TC003. Omit for --all."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 10 test cases."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test cases and exit."
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save a JSON report to reports/."
    )
    args = parser.parse_args()

    console.print("\n[bold cyan]╔══ LLM EVAL FRAMEWORK — BSFI CRM PLAYGROUND ══╗[/bold cyan]\n")

    if args.list:
        list_cases()
        return

    config = load_config()

    # Determine which cases to run
    if args.all:
        test_cases = load_test_cases()
    elif args.cases:
        test_cases = [get_case_by_id(cid) for cid in args.cases]
    else:
        parser.print_help()
        console.print("\n[yellow]Tip: run a case with: python playground.py TC001[/yellow]\n")
        return

    # Run all selected cases
    all_pipeline_outputs   = []
    all_threshold_results  = []

    for test_case in test_cases:
        try:
            result = run_case(test_case, config, verbose=True)
            all_pipeline_outputs.append(result["pipeline_output"])
            all_threshold_results.append(result["threshold_result"])
        except ValueError as e:
            console.print(f"[red]✗ SKIPPED {test_case.get('id', '?')}: {e}[/red]")
        console.print()

    # Release gate
    gate = evaluate_gate(all_threshold_results, config)

    gate_colour = "green" if gate["passed"] else "red"
    console.print(Panel(
        f"[{gate_colour}]{gate['message']}[/{gate_colour}]\n"
        f"Cases passed: {gate['cases_passed']} / {gate['total_cases']}",
        border_style=gate_colour,
        title="Release Gate"
    ))

    if not gate["passed"]:
        console.print("\n[red bold]Critical failures:[/red bold]")
        for f in gate["critical_failures"]:
            rprint(
                f"  [red]✗[/red] {f['case_id']} / {f['metric']}"
                f" — score={f['score']} threshold={f['threshold']}"
            )

    # Optionally save JSON report — runs in finally so it saves even if evaluators threw
    if args.save_report:
        try:
            from src.reporting.report_generator import save_json_report
            path = save_json_report(all_pipeline_outputs, all_threshold_results, gate)
            rprint(f"\n[dim]Report saved: {path}[/dim]")
        except Exception as e:
            rprint(f"\n[red]Report save failed: {e}[/red]")

    console.print()


if __name__ == "__main__":
    main()
