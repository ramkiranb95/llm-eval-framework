"""
ragas_evaluator.py
------------------
STATUS: Tier 2 — experimental, not integrated into Tier 1 pipeline.
        Implements all 4 RAGAs metrics: faithfulness, answer_relevance,
        context_precision, context_recall.
        Use combined_evaluator.py for Tier 1 runs (mode: "combined" in config.yaml).

RAGAs-based evaluator using RAGAs 0.4.x modern API (collections metrics + llm_factory)
routed through an OpenAI-compatible endpoint (Gemini, Groq, or Ollama).

Why this approach works where the old one failed:
    Old path: LangchainLLMWrapper → deprecated, small models fail structured prompts
    New path: AsyncOpenAI client → provider /v1 endpoint → llm_factory → InstructorLLM
    InstructorLLM is what RAGAs collections metrics require.

Metrics:
    faithfulness      — all claims in reply traceable to retrieved context
    answer_relevance  — reply addresses the customer's question
    context_precision — retrieved chunks are relevant (no noise)
    context_recall    — all necessary context was retrieved

context_precision requires: user_input, retrieved_contexts, reference (ground truth)
context_recall    requires: retrieved_contexts, reference (ground truth)

Score extraction: result.value (not .score — RAGAs 0.4.x MetricResult API)

Usage:
    from src.evaluators.experimental.ragas_evaluator import evaluate

Standalone test:
    python -m src.evaluators.experimental.ragas_evaluator
"""

import asyncio
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate(pipeline_output: dict, config: dict) -> dict:
    """
    Run all 4 RAGAs metrics on one pipeline output.

    Args:
        pipeline_output : dict from crm_responder.generate_response()
            Must contain: input_email, generated_reply, retrieved_context,
                          ground_truth (with expected_reply)
        config          : full config dict from load_config()

    Returns:
        {
            "faithfulness"      : {"score": float|None, "error": str|None},
            "answer_relevance"  : {"score": float|None, "error": str|None},
            "context_precision" : {"score": float|None, "error": str|None},
            "context_recall"    : {"score": float|None, "error": str|None},
        }
        score is None if the judge LLM failed.
    """
    from ragas.metrics.collections.faithfulness import Faithfulness
    from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
    from ragas.metrics.collections.context_precision import ContextPrecision
    from ragas.metrics.collections.context_recall import ContextRecall
    from src.utils.config_loader import get_judge_config

    from src.utils.config_loader import get_rag_config
    judge_cfg      = get_judge_config(config)
    model          = judge_cfg["model"]
    api_key        = judge_cfg["api_key"]
    embedding_model = get_rag_config(config).get("embedding_model", "all-MiniLM-L6-v2")
    user_input = pipeline_output["input_email"]
    response   = pipeline_output["generated_reply"]
    contexts   = pipeline_output["retrieved_context"]
    reference  = pipeline_output.get("ground_truth", {}).get("expected_reply", "")

    def build_llm_and_emb():
        """
        Return (llm, emb, async_client) for RAGAs metrics.

        RAGAs 0.4.x collections metrics ONLY accept InstructorLLM — created via
        llm_factory(). All providers go through AsyncOpenAI + llm_factory.

        Gemini:        AsyncOpenAI → Gemini OpenAI-compatible endpoint (/v1beta/openai/)
        Groq / Ollama: AsyncOpenAI → respective OpenAI-compatible endpoint
        """
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import embedding_factory
        client = AsyncOpenAI(base_url=judge_cfg["base_url"], api_key=api_key)
        llm    = llm_factory(model, client=client)
        emb    = embedding_factory("openai", model=embedding_model, client=client)
        return llm, emb, client

    async def _run_all():
        llm, emb, async_client = build_llm_and_emb()
        scores = {}

        try:
            f_metric   = Faithfulness(llm=llm)
            ar_metric  = AnswerRelevancy(llm=llm, embeddings=emb)
            cp_metric  = ContextPrecision(llm=llm)
            cr_metric  = ContextRecall(llm=llm)

            # ── Faithfulness ──────────────────────────────────────────────────
            try:
                result = await f_metric.ascore(
                    user_input=user_input,
                    response=response,
                    retrieved_contexts=contexts
                )
                scores["faithfulness"] = {"score": round(float(result.value), 4), "error": None}
            except Exception as e:
                scores["faithfulness"] = {"score": None, "error": f"RAGAs: {type(e).__name__}: {str(e)[:120]}"}

            # ── Answer Relevance ──────────────────────────────────────────────
            try:
                result = await ar_metric.ascore(
                    user_input=user_input,
                    response=response
                )
                scores["answer_relevance"] = {"score": round(float(result.value), 4), "error": None}
            except Exception as e:
                scores["answer_relevance"] = {"score": None, "error": f"RAGAs: {type(e).__name__}: {str(e)[:120]}"}

            # ── Context Precision ─────────────────────────────────────────────
            # Measures signal-to-noise in retrieved chunks: are the top-ranked
            # chunks actually relevant to the ground truth answer?
            try:
                result = await cp_metric.ascore(
                    user_input=user_input,
                    retrieved_contexts=contexts,
                    reference=reference
                )
                scores["context_precision"] = {"score": round(float(result.value), 4), "error": None}
            except Exception as e:
                scores["context_precision"] = {"score": None, "error": f"RAGAs: {type(e).__name__}: {str(e)[:120]}"}

            # ── Context Recall ────────────────────────────────────────────────
            # Measures coverage: did the retriever surface all facts needed to
            # produce the expected answer?
            try:
                result = await cr_metric.ascore(
                    user_input=user_input,
                    retrieved_contexts=contexts,
                    reference=reference
                )
                scores["context_recall"] = {"score": round(float(result.value), 4), "error": None}
            except Exception as e:
                scores["context_recall"] = {"score": None, "error": f"RAGAs: {type(e).__name__}: {str(e)[:120]}"}

        except Exception as e:
            err = f"Failed to initialise RAGAs metrics: {e}"
            scores = {
                "faithfulness"      : {"score": None, "error": err},
                "answer_relevance"  : {"score": None, "error": err},
                "context_precision" : {"score": None, "error": err},
                "context_recall"    : {"score": None, "error": err},
            }
        finally:
            if async_client is not None:
                await async_client.close()

        return scores

    return asyncio.run(_run_all())


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from src.utils.config_loader import load_config
    from src.utils.data_loader import get_case_by_id
    from src.pipeline.crm_responder import generate_response

    console = Console()
    console.print("\n[bold cyan]╔══ RAGAS EVALUATOR — VERIFICATION (TC001) ══╗[/bold cyan]\n")

    config    = load_config()
    test_case = get_case_by_id("TC001")

    console.print("  → Running CRM responder...")
    result = generate_response(test_case, config)

    console.print("  → Running RAGAs evaluation (all 4 metrics)...\n")
    scores = evaluate(result, config)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric",  min_width=22)
    table.add_column("Score",   min_width=10)
    table.add_column("Status",  min_width=50)

    for metric, data in scores.items():
        score = data["score"]
        error = data["error"]
        if score is not None:
            colour     = "green" if score >= 0.8 else ("yellow" if score >= 0.5 else "red")
            score_str  = f"[{colour}]{score:.4f}[/{colour}]"
            status_str = "[green]✓ scored[/green]"
        else:
            score_str  = "[dim]None[/dim]"
            status_str = f"[yellow]⚠ {(error or '')[:48]}[/yellow]"
        table.add_row(metric, score_str, status_str)

    console.print(table)
    console.print("\n[bold green]✓ RAGAs evaluator done[/bold green]\n")
