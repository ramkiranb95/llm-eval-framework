"""
playground.py
-------------
The local BSFI hands-on sandbox. Run any single test case or all 16 instantly.

This is your fast feedback loop — no pytest, no full report file.
Use this to:
  - Try a new test case and see all scores immediately
  - Tune thresholds in config.yaml and see the impact
  - Observe which LLM syndromes appear in which cases
  - Understand what RAGAs / DeepEval are actually measuring

Usage:
    python playground.py TC001          # run single case
    python playground.py TC003 TC008    # run specific cases
    python playground.py --all          # run all 16 cases
    python playground.py --list         # list available cases
"""

import sys
import time
import argparse
import concurrent.futures
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import print as rprint

from src.utils.config_loader import load_config, get_pipeline_config
from src.utils.data_loader import load_test_cases, get_case_by_id
from src.pipeline.crm_responder import generate_response
from src.evaluators.custom_evaluator import evaluate as custom_evaluate, language_check
from src.scoring.threshold_checker import check_thresholds
from src.scoring.release_gate import evaluate_gate

# DeepEval key → canonical metric name used throughout the framework
_DE_ALIAS = {
    "answer_relevancy": "answer_relevance",
}


console = Console()


# ── Run a single case ─────────────────────────────────────────────────────────

def run_case(test_case: dict, config: dict, verbose: bool = True) -> dict:
    """
    Run one test case through the full evaluation pipeline.

    Pre-conditions checked before any LLM call:
      1. email_body must be >= min_email_body_length (config.yaml pipeline section)
      2. SUT output must be valid (non-empty reply, no meta_parse_error, non-empty context)

    Returns:
        {
            "pipeline_output"   : dict from crm_responder
            "all_scores"        : merged scores from all evaluators
            "threshold_result"  : dict from threshold_checker (with "id")
        }

    Raises:
        ValueError if pre-conditions are not met (caught and logged by main()).
    """
    pipeline_cfg = get_pipeline_config(config)
    case_id      = test_case["id"]

    if verbose:
        console.print(Rule(f"[bold cyan]{case_id} — {test_case['intent']}[/bold cyan]"))
        console.print(f"[bold]Category:[/bold] {test_case['category']}  |  [bold]Priority:[/bold] {test_case['priority']}")
        console.print(f"[bold]Email:[/bold]    {test_case['email_subject']}")
        console.print(f"[dim]Syndrome watch: {test_case.get('llm_syndrome_watch', '—')}[/dim]\n")

    # ── Gate 1: Pre-LLM checks ───────────────────────────────────────────────
    email_body = test_case.get("email_body", "")

    # 1a. Minimum length
    min_length = pipeline_cfg["min_email_body_length"]
    if len(email_body) < min_length:
        raise ValueError(
            f"email body too short ({len(email_body)} chars, minimum {min_length}) "
            f"— skipped before LLM call"
        )

    # 1b. Input language check — non-English email skips LLM, routes to human
    lang_threshold = pipeline_cfg["language_check_ascii_threshold"]
    lang_result    = language_check(email_body, lang_threshold)
    if lang_result["score"] == 0.0:
        raise ValueError(
            f"non-English input detected — skipped before LLM call "
            f"({lang_result['notes']})"
        )

    # ── Step 1: Generate CRM reply (with per-case timeout) ────────────────────
    case_timeout = pipeline_cfg["case_timeout_seconds"]

    if verbose:
        console.print(f"[dim]Running CRM responder (timeout: {case_timeout}s)...[/dim]")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_response, test_case, config, verbose)
            pipeline_output = future.result(timeout=case_timeout)
    except concurrent.futures.TimeoutError:
        raise ValueError(
            f"generate_response() timed out after {case_timeout}s — "
            f"Ollama may be unresponsive"
        )

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
    custom_scores = custom_evaluate(pipeline_output, config=config)

    # ── Gate 2: Confidence gate — skip LLM eval if SUT output is invalid ─────
    meta_parse_error = pipeline_output.get("meta_parse_error")
    empty_context    = len(pipeline_output.get("retrieved_context", [])) == 0

    if meta_parse_error or empty_context:
        reason = "meta_parse_error" if meta_parse_error else "empty retrieved context"
        if verbose:
            console.print(f"[yellow]⚠ LLM evaluation skipped — SUT output invalid ({reason})[/yellow]")
        llm_scores = _skipped_llm_scores(reason)
    elif eval_mode == "combined":
        from src.evaluators.combined_evaluator import evaluate as combined_evaluate
        llm_scores = combined_evaluate(pipeline_output, config)
    else:
        # Separate mode — call RAGAs and DeepEval libraries independently,
        # then merge into the same flat dict structure as combined mode.
        from src.evaluators.experimental.ragas_evaluator import evaluate as ragas_evaluate
        from src.evaluators.experimental.deepeval_evaluator import evaluate as deepeval_evaluate
        from src.evaluators.combined_evaluator import LLM_METRICS

        ragas_scores   = ragas_evaluate(pipeline_output, config)
        deepeval_raw   = deepeval_evaluate(pipeline_output, config)

        # Flatten DeepEval results with alias normalisation
        deepeval_scores: dict = {}
        for k, v in deepeval_raw.items():
            canonical = _DE_ALIAS.get(k, k)
            deepeval_scores[canonical] = v

        # Merge: RAGAs takes precedence for overlapping metrics (faithfulness, answer_relevance)
        # because RAGAs uses multi-step claim decomposition — more rigorous than single-pass.
        _not_implemented = {"score": None, "error": "not available in separate mode — use combined"}
        llm_scores = {}
        for metric in LLM_METRICS:
            if metric in ragas_scores:
                llm_scores[metric] = ragas_scores[metric]
            elif metric in deepeval_scores:
                llm_scores[metric] = deepeval_scores[metric]
            else:
                llm_scores[metric] = _not_implemented

    # Merge all scores into one flat dict
    all_scores = {}
    all_scores.update(custom_scores)
    all_scores.update(llm_scores)

    # ── Routing decision log (after all scores available) ────────────────────
    if verbose:
        _print_routing_decision(pipeline_output, custom_scores, llm_scores, config)

    # ── Step 3: Threshold check ──────────────────────────────────────────────
    threshold_result = check_thresholds(all_scores, config)
    threshold_result["id"] = case_id

    # ── Step 4: Print scores table ───────────────────────────────────────────
    if verbose:
        _print_scores_table(threshold_result)

    return {
        "pipeline_output"  : pipeline_output,
        "all_scores"       : all_scores,
        "threshold_result" : threshold_result
    }


def _skipped_llm_scores(reason: str) -> dict:
    """Return null scores for all LLM-based metrics when the confidence gate fires."""
    from src.evaluators.combined_evaluator import LLM_METRICS
    error_msg = f"skipped — SUT output invalid ({reason})"
    scores    = {}
    for metric in LLM_METRICS:
        scores[metric] = {"score": None, "error": error_msg}
        if metric == "hallucination":
            scores[metric]["reason"] = ""
    return scores


def _print_routing_decision(
    pipeline_output: dict,
    custom_scores: dict,
    llm_scores: dict,
    config: dict,
) -> None:
    """
    Print a routing decision block after all evaluators have run.

    Three classes of triggers:
      1. Deterministic — language_check, restricted_words (from custom_evaluator)
      2. SUT prediction — predicted_escalation=True
      3. LLM confidence — any critical LLM metric score below pipeline.confidence_threshold
         (inverted metrics use 1 - confidence_threshold as their upper bound)
    """
    from src.evaluators.combined_evaluator import INVERTED_METRICS
    from src.scoring.threshold_checker import _is_critical

    confidence_threshold = config.get("pipeline", {}).get("confidence_threshold", 0.50)
    reasons = []

    # ── 1. Language check ────────────────────────────────────────────────────
    lang = custom_scores.get("language_check", {})
    if lang.get("score") == 0.0:
        reasons.append(f"language_check=0.0 — {lang.get('notes', 'non-English detected')}")

    # ── 2. Escalation predicted by SUT ──────────────────────────────────────
    if pipeline_output.get("predicted_escalation"):
        reasons.append(
            f"escalation=True — {pipeline_output.get('ticket_reasoning', 'SUT flagged for escalation')}"
        )

    # ── 3. Restricted words in reply ─────────────────────────────────────────
    rw = custom_scores.get("restricted_words", {})
    if rw.get("score") == 0.0:
        reasons.append(f"restricted_words=0.0 — {rw.get('notes', 'violation found')}")

    # ── 4. LLM confidence threshold — critical metrics only ──────────────────
    for metric, data in llm_scores.items():
        score = data.get("score")
        if score is None:
            continue

        if not _is_critical(metric, config):
            continue

        inverted = metric in INVERTED_METRICS
        if inverted:
            # lower is better — flag if score exceeds (1 - confidence_threshold)
            upper_bound = 1.0 - confidence_threshold
            if score > upper_bound:
                reasons.append(
                    f"{metric}={score:.3f} ↑ above {upper_bound:.2f} (confidence_threshold={confidence_threshold})"
                )
        else:
            # higher is better — flag if score is below confidence_threshold
            if score < confidence_threshold:
                reasons.append(
                    f"{metric}={score:.3f} below confidence_threshold={confidence_threshold}"
                )

    # ── Composite confidence score — weighted average of critical LLM metrics ──
    # Weights reflect BSFI risk priority: hallucination + faithfulness matter most
    CONFIDENCE_WEIGHTS = {
        "faithfulness"      : 0.30,
        "hallucination"     : 0.30,   # inverted — contribution = 1 - score
        "answer_relevance"  : 0.20,
        "answer_correctness": 0.20,
    }
    weighted_sum  = 0.0
    weight_total  = 0.0
    for metric, weight in CONFIDENCE_WEIGHTS.items():
        score = llm_scores.get(metric, {}).get("score")
        if score is None:
            continue
        contribution = (1.0 - score) if metric in INVERTED_METRICS else score
        weighted_sum  += contribution * weight
        weight_total  += weight

    confidence_score = (weighted_sum / weight_total) if weight_total > 0 else None

    ticket_status = pipeline_output.get("predicted_ticket_status", "—")

    if reasons:
        console.print(f"\n[bold red]⚠  ROUTING: Assign to human agent[/bold red]")
        console.print(f"   Ticket status : [yellow]{ticket_status}[/yellow]")
        if confidence_score is not None:
            console.print(f"   Confidence    : [yellow]{confidence_score:.2f}[/yellow] (threshold: {confidence_threshold})")
        for r in reasons:
            console.print(f"   Reason        : [yellow]{r}[/yellow]")
    else:
        console.print(f"\n[bold green]✓  ROUTING: Auto-reply eligible[/bold green]")
        console.print(f"   Ticket status : [green]{ticket_status}[/green]")
        if confidence_score is not None:
            console.print(f"   Confidence    : [green]{confidence_score:.2f}[/green] (threshold: {confidence_threshold})")


def _print_scores_table(threshold_result: dict) -> None:
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

        if score is None:
            colour = "yellow"
        elif passed is True:
            colour = "green"
        elif passed is False and critical:
            colour = "red"
        else:
            colour = "yellow"

        table.add_row(
            metric,
            f"[{colour}]{score:.3f}[/{colour}]" if score is not None else f"[{colour}]—[/{colour}]",
            f"{threshold:.2f}" if threshold is not None else "—",
            f"[{colour}]{'✓' if passed else '✗' if passed is False else '?'}[/{colour}]",
            "[red]YES[/red]" if critical else "no",
            (f"↓ inverted  " if inverted else "") + (str(error)[:36] if error else "")
        )

    console.print(table)


# ── List available cases ──────────────────────────────────────────────────────

def list_cases() -> None:
    cases = load_test_cases()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID",       width=7)
    table.add_column("Category", width=14)
    table.add_column("Intent",   width=30)
    table.add_column("Priority", width=10)
    table.add_column("Escalate", width=9)
    for c in cases:
        table.add_row(
            c["id"], c["category"], c["intent"], c["priority"],
            str(c["expected_escalation"])
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
        help="Run all test cases."
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

    config       = load_config()
    pipeline_cfg = get_pipeline_config(config)

    # Determine which cases to run
    if args.all:
        test_cases = load_test_cases()
    elif args.cases:
        test_cases = [get_case_by_id(cid) for cid in args.cases]
    else:
        parser.print_help()
        console.print("\n[yellow]Tip: run a case with: python playground.py TC001[/yellow]\n")
        return

    all_pipeline_outputs  = []
    all_threshold_results = []
    delay                 = pipeline_cfg["inter_case_delay_seconds"]

    for i, test_case in enumerate(test_cases):
        try:
            result = run_case(test_case, config, verbose=True)
            all_pipeline_outputs.append(result["pipeline_output"])
            all_threshold_results.append(result["threshold_result"])
        except ValueError as e:
            console.print(f"[red]✗ SKIPPED {test_case.get('id', '?')}: {e}[/red]")
        console.print()

        # Inter-case delay on batch runs to avoid hitting rate limits
        if args.all and i < len(test_cases) - 1 and delay > 0:
            console.print(f"[dim]Waiting {delay}s before next case...[/dim]")
            time.sleep(delay)

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

    console.print("\n[bold]Metric Reasons:[/bold]")
    for cr in all_threshold_results:
        case_id = cr.get("id", "—")
        console.print(f"\n  [bold cyan]{case_id}[/bold cyan]")
        for metric, data in cr.items():
            if metric == "id" or not isinstance(data, dict):
                continue
            reason = data.get("reason", "")
            if not reason:
                continue
            passed = data.get("passed", False)
            colour = "green" if passed else "red"
            symbol = "✓" if passed else "✗"
            score  = data.get("score")
            score_str = f"{score:.2f}" if score is not None else "—"
            rprint(f"    [{colour}]{symbol}[/{colour}] [bold]{metric}[/bold] ({score_str}): {reason}")

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
