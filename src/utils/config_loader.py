"""
config_loader.py
----------------
Single source of truth for all configuration in the framework.
Every module imports from here — nothing reads config.yaml directly.

Usage:
    from src.utils.config_loader import load_config, get_llm_config

Standalone test:
    python -m src.utils.config_loader
"""

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path


# ── Resolve project root (works from any working directory) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH     = PROJECT_ROOT / ".env"


def load_env() -> None:
    """Load .env file into environment variables. Silent if file doesn't exist."""
    load_dotenv(dotenv_path=ENV_PATH)


def validate_config_thresholds(config: dict) -> None:
    """
    Check every metric threshold in config["evaluation"] is a float in [0.0, 1.0].
    Raises ValueError with a list of all bad values so the user can fix them all at once.
    Called by load_config() — fails before any case or LLM call is made.
    """
    errors = []
    for group, metrics in config.get("evaluation", {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, cfg in metrics.items():
            if not isinstance(cfg, dict):
                continue
            threshold = cfg.get("threshold")
            if threshold is None:
                errors.append(f"  {group}.{metric_name}: 'threshold' key is missing")
            elif not isinstance(threshold, (int, float)):
                errors.append(f"  {group}.{metric_name}: threshold must be a number, got {threshold!r}")
            elif not (0.0 <= float(threshold) <= 1.0):
                errors.append(f"  {group}.{metric_name}: threshold {threshold} is out of range [0.0, 1.0]")

    if errors:
        raise ValueError(
            "config.yaml has invalid threshold values — fix before running:\n"
            + "\n".join(errors)
        )


def load_config(path: Path = CONFIG_PATH) -> dict:
    """
    Load and return the full config.yaml as a dict.

    Args:
        path: Path to config file. Defaults to config/config.yaml.

    Returns:
        dict: Full configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If any metric threshold is missing or out of range [0.0, 1.0].
    """
    load_env()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    validate_config_thresholds(config)

    return config


# ── Helper getters — use these in other modules ──────────────────────────────

def get_llm_config(config: dict = None) -> dict:
    """Returns the full llm config block. Use get_sut_config() or get_judge_config() instead."""
    config = config or load_config()
    return config["llm"]


def get_sut_config(config: dict = None) -> dict:
    """
    Returns config for the System Under Test (CRM responder).
    Always runs on local Ollama — this is the model being evaluated.

    Keys:
        sut_model       : e.g. "mistral"
        ollama_base_url : "http://localhost:11434"
        temperature     : 0.0
    """
    config = config or load_config()
    llm    = config["llm"]
    return {
        "model"      : llm["sut_model"],
        "base_url"   : llm["ollama_base_url"],
        "temperature": llm["temperature"]
    }


def get_judge_config(config: dict = None) -> dict:
    """
    Returns config for the judge LLM (used by RAGAs and DeepEval evaluators).
    Provider is set in config.yaml → llm.judge_provider.
    API key is read from .env automatically.

    Returns:
        {
            "provider" : "gemini" | "groq" | "ollama",
            "model"    : model name string,
            "base_url" : OpenAI-compatible endpoint URL,
            "api_key"  : API key from .env (or "ollama" for local),
        }

    Raises:
        ValueError if the API key for the configured provider is not set in .env
    """
    load_env()
    config   = config or load_config()
    llm      = config["llm"]
    provider = llm.get("judge_provider", "ollama")
    model    = llm.get("judge_model", llm.get("sut_model", "mistral"))

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "paste_your_gemini_key_here":
            raise ValueError(
                "GEMINI_API_KEY is not set in .env — "
                "add your key or switch judge_provider to 'groq' or 'ollama'"
            )
        return {
            "provider" : "gemini",
            "model"    : model,
            "base_url" : "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key"  : api_key,
        }

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key == "paste_your_groq_key_here":
            raise ValueError(
                "GROQ_API_KEY is not set in .env — "
                "add your key or switch judge_provider to 'gemini' or 'ollama'"
            )
        return {
            "provider" : "groq",
            "model"    : model,
            "base_url" : "https://api.groq.com/openai/v1",
            "api_key"  : api_key,
        }

    else:  # ollama — local fallback
        return {
            "provider" : "ollama",
            "model"    : model,
            "base_url" : llm["ollama_base_url"] + "/v1",
            "api_key"  : "ollama",
        }


def get_rag_config(config: dict = None) -> dict:
    """
    Returns RAG pipeline configuration.

    Keys:
        vector_db       : "chromadb"
        embedding_model : "all-MiniLM-L6-v2"
        top_k           : number of context chunks to retrieve (1-10)
        documents_path  : path to policy docs folder
        chunk_size      : token size per chunk (100-1000)
        chunk_overlap   : overlap between chunks (0-200)
    """
    config = config or load_config()
    return config["rag"]


def get_metrics_config(config: dict = None) -> dict:
    """
    Returns all evaluation metric configurations.

    Structure:
        ragas:
            faithfulness:    {enabled, threshold, critical, description}
            answer_relevance:{enabled, threshold, critical, description}
            context_precision:{...}
            context_recall:  {...}
        deepeval:
            hallucination:   {enabled, threshold, critical, description}
            answer_correctness:{...}
            coherence:       {...}
        custom:
            ticket_status_accuracy:{...}
            escalation_logic:{...}
            key_facts_coverage:{...}
            out_of_scope_handling:{...}

    Threshold ranges:
        Most metrics  : 0.0 – 1.0 (higher = stricter, must EXCEED threshold to pass)
        hallucination : 0.0 – 1.0 (lower = better, must STAY BELOW threshold to pass)
    """
    config = config or load_config()
    return config["evaluation"]


def get_test_data_config(config: dict = None) -> dict:
    """
    Returns test data configuration.

    Keys:
        ground_truth_included : bool
        run_all_cases         : bool
        specific_case_ids     : list of IDs to run (empty = run all)
    """
    config = config or load_config()
    return config["test_data"]


def get_reporting_config(config: dict = None) -> dict:
    """
    Returns reporting configuration.

    Keys:
        output_path              : folder where reports are saved
        formats                  : ["json"]
        include_traceability     : bool
        include_per_case_breakdown: bool
        timestamp_reports        : bool
    """
    config = config or load_config()
    return config["reporting"]


def get_release_gate_config(config: dict = None) -> dict:
    """
    Returns release gate configuration.

    Keys:
        enabled         : bool
        policy          : "all_critical_must_pass" | "majority_must_pass"
        block_on_failure: bool
        summary_message : {pass: str, fail: str}
    """
    config = config or load_config()
    return config["release_gate"]


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint

    console = Console()
    config  = load_config()

    console.print("\n[bold cyan]╔══ CONFIG LOADER — VERIFICATION ══╗[/bold cyan]\n")

    # Project info
    rprint(f"[bold]Project:[/bold] {config['project']['name']}")
    rprint(f"[bold]Version:[/bold] {config['project']['version']}")
    rprint(f"[bold]Tier:[/bold]    {config['project']['tier']}\n")

    # LLM config
    llm = get_llm_config(config)
    console.print("[bold yellow]── LLM Configuration ──[/bold yellow]")
    rprint(f"  SUT model       : {llm['sut_model']}")
    rprint(f"  Judge provider  : {llm['judge_provider']}")
    rprint(f"  Judge model     : {llm['judge_model']}")
    rprint(f"  Ollama URL      : {llm['ollama_base_url']}")
    rprint(f"  Temperature     : {llm['temperature']}\n")

    # Metrics table
    console.print("[bold yellow]── Evaluation Metrics ──[/bold yellow]")
    metrics = get_metrics_config(config)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric",    style="cyan",  width=28)
    table.add_column("Group",     style="white", width=10)
    table.add_column("Threshold", style="green", width=10)
    table.add_column("Critical",  style="red",   width=10)
    table.add_column("Enabled",   style="white", width=8)

    for group, group_metrics in metrics.items():
        for metric_name, cfg in group_metrics.items():
            table.add_row(
                metric_name,
                group,
                str(cfg["threshold"]),
                "YES" if cfg["critical"] else "no",
                "✓" if cfg["enabled"] else "✗"
            )

    console.print(table)

    # Release gate
    gate = get_release_gate_config(config)
    console.print(f"\n[bold yellow]── Release Gate ──[/bold yellow]")
    rprint(f"  Policy  : {gate['policy']}")
    rprint(f"  Enabled : {gate['enabled']}")
    rprint(f"  Message : {gate['summary_message']['pass']}")

    console.print("\n[bold green]✓ Config loaded successfully[/bold green]\n")
