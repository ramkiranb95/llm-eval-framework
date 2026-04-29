"""
deepeval_evaluator.py
---------------------
STATUS: Tier 2 — experimental, not integrated into Tier 1 pipeline.
        Only implements hallucination (1 of 3 DeepEval metrics).
        answer_correctness and coherence are not yet implemented.
        Use combined_evaluator.py for Tier 1 runs (mode: "combined" in config.yaml).

DeepEval-based evaluator for hallucination detection.

Uses a custom OllamaJudge wrapper to connect DeepEval's HallucinationMetric
to a local Ollama LLM — no OpenAI key required.

Tier 1 active metric:
    hallucination — does the reply contain claims NOT supported by the retrieved context?

Score interpretation (inverted from other metrics):
    0.0 = no hallucination (best)
    1.0 = fully hallucinated (worst)
Threshold: score <= hallucination_threshold means PASS.

Usage:
    from src.evaluators.experimental.deepeval_evaluator import evaluate

Standalone test:
    python -m src.evaluators.experimental.deepeval_evaluator
"""

import asyncio
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI


# ── Generic OpenAI-compatible wrapper for DeepEval ───────────────────────────

class OpenAICompatibleJudge(DeepEvalBaseLLM):
    """
    Wraps any OpenAI-compatible endpoint as a DeepEval judge.
    Works with Groq, Ollama /v1, and other OpenAI-compatible APIs.
    DeepEval requires generate() and a_generate() — we implement both.
    """

    def __init__(self, model_name: str, base_url: str, api_key: str, temperature: float = 0.0):
        self._model_name  = model_name
        self._base_url    = base_url
        self._api_key     = api_key
        self._temperature = temperature
        # Note: don't call super().__init__() — it calls load_model() which we override

    def load_model(self):
        return ChatOpenAI(
            model=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            temperature=self._temperature
        )

    def generate(self, prompt: str, *args, **kwargs) -> str:
        llm = self.load_model()
        return llm.invoke(prompt).content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model_name


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate(pipeline_output: dict, config: dict) -> dict:
    """
    Run DeepEval HallucinationMetric on one pipeline output.

    Args:
        pipeline_output : dict from crm_responder.generate_response()
            Must contain: input_email, generated_reply, retrieved_context,
                          ground_truth (with expected_reply)
        config          : full config dict from load_config()

    Returns:
        {
            "hallucination": {"score": float|None, "error": str|None, "reason": str}
        }
        score is None if the judge LLM failed.
        Lower score = less hallucination = better.
    """
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase
    from src.utils.config_loader import get_judge_config

    judge_cfg = get_judge_config(config)
    provider  = judge_cfg["provider"]

    if provider == "gemini":
        from deepeval.models import GeminiModel
        judge = GeminiModel(model=judge_cfg["model"], api_key=judge_cfg["api_key"])
    else:
        # groq and ollama — both expose an OpenAI-compatible endpoint
        # judge_cfg already has the correct base_url and api_key for each provider
        judge = OpenAICompatibleJudge(
            model_name  = judge_cfg["model"],
            base_url    = judge_cfg["base_url"],
            api_key     = judge_cfg["api_key"],
            temperature = 0.0
        )

    test_case = LLMTestCase(
        input          = pipeline_output["input_email"],
        actual_output  = pipeline_output["generated_reply"],
        expected_output= pipeline_output["ground_truth"]["expected_reply"],
        context        = pipeline_output["retrieved_context"]
    )

    metric = HallucinationMetric(
        threshold = 0.5,    # Internal threshold for DeepEval's own pass/fail
        model     = judge
    )

    try:
        metric.measure(test_case)
        score = metric.score
        reason = getattr(metric, "reason", "")
        return {
            "hallucination": {
                "score" : round(float(score), 4) if score is not None else None,
                "error" : None,
                "reason": reason or ""
            }
        }
    except Exception as e:
        return {
            "hallucination": {
                "score" : None,
                "error" : f"DeepEval judge failed: {type(e).__name__}: {str(e)[:120]}",
                "reason": ""
            }
        }


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    from src.utils.config_loader import load_config
    from src.utils.data_loader import get_case_by_id
    from src.pipeline.crm_responder import generate_response

    console = Console()
    console.print("\n[bold cyan]╔══ DEEPEVAL EVALUATOR — VERIFICATION (TC001) ══╗[/bold cyan]\n")
    console.print("[dim]Note: HallucinationMetric score is inverted — 0.0 means no hallucination (best).\nScore <= hallucination_threshold is a PASS.[/dim]\n")

    config    = load_config()
    test_case = get_case_by_id("TC001")

    console.print("  → Running CRM responder...")
    result = generate_response(test_case, config)

    console.print("  → Running DeepEval hallucination check...\n")
    scores = evaluate(result, config)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric",       width=16)
    table.add_column("Score",        width=10)
    table.add_column("Status",       width=30)
    table.add_column("Reason",       width=40)

    from src.utils.config_loader import get_metrics_config
    hallucination_threshold = get_metrics_config(config)["deepeval"]["hallucination"]["threshold"]

    for metric, data in scores.items():
        score  = data["score"]
        error  = data["error"]
        reason = data.get("reason", "")

        if score is not None:
            passed = score <= hallucination_threshold
            colour = "green" if passed else "red"
            score_str  = f"[{colour}]{score:.4f}[/{colour}]"
            status_str = f"[{colour}]{'✓ PASS' if passed else '✗ FAIL'} (threshold ≤ {hallucination_threshold})[/{colour}]"
        else:
            score_str  = "[dim]None[/dim]"
            status_str = f"[yellow]⚠ {error}[/yellow]"
            reason     = ""

        table.add_row(metric, score_str, status_str, reason[:38] if reason else "—")

    console.print(table)

    console.print("\n[bold green]✓ DeepEval evaluator working[/bold green]\n")
