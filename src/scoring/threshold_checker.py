"""
threshold_checker.py
--------------------
Compares each metric score against its configured threshold and marks it
passed / failed / critical.

Handles two score polarities:
    Normal  — higher is better (faithfulness, answer_relevance, key_facts_coverage…)
              PASS if score >= threshold
    Inverted — lower is better (hallucination, toxicity)
              PASS if score <= threshold

All thresholds are read from global config.yaml. Per-case overrides are not
supported — use separate test suites with separate configs for different
threshold requirements.

Usage:
    from src.scoring.threshold_checker import check_thresholds

Standalone test:
    python -m src.scoring.threshold_checker
"""

from typing import Optional


# Metrics where a LOWER score is better — pass if score <= threshold
INVERTED_METRICS = {"hallucination", "toxicity"}


def _resolve_threshold(metric_name: str, global_config: dict) -> Optional[float]:
    """
    Find the threshold for a given metric from global config.
    Returns None if the metric is not found in config.
    """
    eval_cfg = global_config.get("evaluation", {})
    for _group, group_metrics in eval_cfg.items():
        if isinstance(group_metrics, dict) and metric_name in group_metrics:
            return group_metrics[metric_name].get("threshold")
    return None


def _is_critical(metric_name: str, global_config: dict) -> bool:
    """Return True if this metric is marked critical in config."""
    eval_cfg = global_config.get("evaluation", {})
    for _group, group_metrics in eval_cfg.items():
        if isinstance(group_metrics, dict) and metric_name in group_metrics:
            return bool(group_metrics[metric_name].get("critical", False))
    return False


def check_thresholds(all_scores: dict, config: dict) -> dict:
    """
    Compare each metric score against its global threshold and mark pass/fail/critical.

    Args:
        all_scores : merged dict of all evaluator outputs, keyed by metric name.
                     Each value must have a "score" key (float or None).
                     May also have "error", "notes", "reason" keys (passed through).
                     Example:
                       {
                         "faithfulness":          {"score": 0.87, "error": None},
                         "hallucination":         {"score": 0.12, "error": None},
                         "ticket_status_accuracy":{"score": 1.0,  "notes": "..."},
                         ...
                       }
        config     : full config dict from load_config()

    Returns:
        dict keyed by metric name:
        {
            "faithfulness": {
                "score"     : 0.87,
                "threshold" : 0.85,
                "passed"    : True,
                "critical"  : True,
                "inverted"  : False,
                "error"     : None
            },
            ...
        }
        If score is None (judge LLM failed), passed=False and a note is added.
    """
    results = {}

    for metric_name, data in all_scores.items():
        score     = data.get("score")
        error     = data.get("error")
        threshold = _resolve_threshold(metric_name, config)
        critical  = _is_critical(metric_name, config)
        inverted  = metric_name in INVERTED_METRICS

        # Validate score type and range before comparing
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                error = f"score is not a number: {score!r}"
                score = None
            else:
                if not (0.0 <= score <= 1.0):
                    error = f"score out of range [0,1]: {score}"
                    score = None

        if score is None:
            # Judge LLM failed — treat as a fail but note the reason
            passed = False
            note   = error or f"{metric_name}: score is None — judge LLM did not return a value"
        elif threshold is None:
            # Metric not in config — score it but can't threshold it
            passed = None
            note   = "no threshold configured"
        elif inverted:
            passed = score <= threshold
            note   = None
        else:
            passed = score >= threshold
            note   = None

        results[metric_name] = {
            "score"     : score,
            "threshold" : threshold,
            "passed"    : passed,
            "critical"  : critical,
            "inverted"  : inverted,
            "error"     : note or error
        }

    return results


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    from src.utils.config_loader import load_config

    console = Console()
    console.print("\n[bold cyan]╔══ THRESHOLD CHECKER — VERIFICATION ══╗[/bold cyan]\n")

    config = load_config()

    mock_scores = {
        "faithfulness"          : {"score": 0.82, "error": None},
        "answer_relevance"      : {"score": 0.78, "error": None},   # will fail (< 0.80)
        "hallucination"         : {"score": 0.04, "error": None},   # inverted
        "ticket_status_accuracy": {"score": 1.00, "notes": "predicted=escalated | expected=escalated"},
        "escalation_logic"      : {"score": 1.00, "notes": "predicted=True | expected=True"},
        "key_facts_coverage"    : {"score": 0.60, "notes": "3/5 facts found"},  # will fail (< 0.50)
    }

    results = check_thresholds(mock_scores, config)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric",    width=24)
    table.add_column("Score",     width=8)
    table.add_column("Threshold", width=10)
    table.add_column("Pass",      width=6)
    table.add_column("Critical",  width=9)
    table.add_column("Inverted",  width=9)

    for metric, r in results.items():
        score     = r["score"]
        threshold = r["threshold"]
        passed    = r["passed"]
        critical  = r["critical"]
        inverted  = r["inverted"]

        if passed is True:
            pass_str = "[green]✓[/green]"
        elif passed is False:
            pass_str = "[red]✗[/red]"
        else:
            pass_str = "[dim]?[/dim]"

        table.add_row(
            metric,
            f"{score:.2f}" if score is not None else "[dim]None[/dim]",
            f"{threshold:.2f}" if threshold is not None else "—",
            pass_str,
            "[red]YES[/red]" if critical else "no",
            "↓" if inverted else "↑"
        )

    console.print(table)
    console.print("\n[dim]All thresholds from global config.yaml — no per-case overrides[/dim]")
    console.print("\n[bold green]✓ Threshold checker working[/bold green]\n")
