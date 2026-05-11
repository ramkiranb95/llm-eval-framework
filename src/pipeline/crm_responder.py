"""
crm_responder.py
----------------
The System Under Test (SUT) — simulates a BSFI CRM Auto-Responder.

Takes a customer email + retrieved context chunks and produces in ONE LLM call:
  1. A professional CRM reply to the customer
  2. Ticket status prediction (open / in_progress / escalated / resolved)
  3. Escalation flag

Single-call design: reply and ticket metadata are requested together using a
delimiter format — reply in free text, metadata as JSON, separated by tags.
This avoids embedding the reply inside JSON (which breaks on quotes/newlines).

Output format the LLM is asked to produce:
    [REPLY]
    Dear <name>, ...
    Regards, Customer Support Team
    [/REPLY]
    [META]
    {"ticket_status": "...", "escalation": true/false, "reasoning": "..."}
    [/META]

This is Tier 1 — RAG is simulated (retrieved_context comes pre-filled
from context.json via data_loader). No real ChromaDB call happens here.

Standalone test:
    python -m src.pipeline.crm_responder
"""

import json
import re
import time
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from src.utils.config_loader import load_config, get_sut_config, get_rag_config


# ── System prompt — defines the persona of the CRM agent ────────────────────

SYSTEM_PROMPT = """You are a professional customer service agent for a BSFI \
(Banking, Small Finance, Micro Lending) company in India.

Your responsibilities:
- Respond to customer emails in a helpful, empathetic, and professional tone
- Use ONLY the provided context to answer — do not make up information
- If the query is outside your scope, politely redirect without fabricating advice
- Always address the customer by name
- End every reply with: "Regards, Customer Support Team"

IMPORTANT: Base your response strictly on the retrieved context provided.
Do not use knowledge outside of what is given to you."""


# ── Prompt builder ───────────────────────────────────────────────────────────

def _build_combined_prompt(email_subject: str, email_body: str, context_chunks: list) -> str:
    """
    Single prompt that asks the LLM to produce both the CRM reply and ticket
    metadata in one call, using delimiter tags to avoid JSON-escaping issues.
    """
    context_text = "\n".join(f"- {chunk}" for chunk in context_chunks)
    return f"""{SYSTEM_PROMPT}

--- RETRIEVED CONTEXT (use only this) ---
{context_text}

--- CUSTOMER EMAIL ---
Subject: {email_subject}
{email_body}

--- OUTPUT FORMAT ---
You must respond using exactly these four tags. Do not skip any tag.
After [/REPLY], you must always output [META] with valid JSON.

Ticket status rules:
- escalated : serious complaints, regulatory issues, financial disputes, mis-selling allegations
- resolved  : query fully answered, no further action needed (simple queries, out-of-scope redirects)
- in_progress: standard queries being processed, callbacks scheduled
- open      : received but not yet actioned

[REPLY]
<write your CRM reply here — use only retrieved context, end with "Regards, Customer Support Team">
[/REPLY]
[META]
{{"ticket_status": "<escalated|resolved|in_progress|open>", "escalation": <true|false>, "reasoning": "<one sentence>"}}
[/META]"""


# ── LLM caller ──────────────────────────────────────────────────────────────

def _get_llm(sut_cfg: dict) -> ChatOpenAI:
    """
    Create and return a ChatOpenAI instance for the configured SUT provider.
    Gemini, Groq, and Ollama all expose an OpenAI-compatible endpoint.
    """
    return ChatOpenAI(
        model       = sut_cfg["model"],
        base_url    = sut_cfg["base_url"],
        api_key     = sut_cfg["api_key"],
        temperature = sut_cfg["temperature"],
    )


def _parse_combined_output(raw_output: str, verbose: bool = True) -> tuple[str, dict]:
    """
    Extract reply and ticket metadata from the combined LLM output.

    Expected format:
        [REPLY]
        <reply text>
        [/REPLY]
        [META]
        {"ticket_status": "...", "escalation": ..., "reasoning": "..."}
        [/META]

    Returns:
        (reply_text, ticket_dict)
    ticket_dict includes "parse_error": True when the META block could not be parsed
    and the fallback default was used. Falls back gracefully if either section is missing.
    """
    # Extract reply — try proper closing tag first, then content between [REPLY] and [META]
    # some SUT models omit closing tags — fallback handles that
    reply_match = re.search(r'\[REPLY\](.*?)\[/REPLY\]', raw_output, re.DOTALL)
    if not reply_match:
        reply_match = re.search(r'\[REPLY\](.*?)(?=\[META\]|\Z)', raw_output, re.DOTALL)
    reply = reply_match.group(1).strip() if reply_match else raw_output.strip()

    # Extract metadata JSON — try proper closing tag first, then to end of string
    meta_match = re.search(r'\[META\](.*?)\[/META\]', raw_output, re.DOTALL)
    if not meta_match:
        meta_match = re.search(r'\[META\](.*?)\Z', raw_output, re.DOTALL)
    if meta_match:
        meta_text = meta_match.group(1).strip()
        # Strip markdown code fences that some models wrap around JSON
        meta_text = re.sub(r'^```(?:json)?\s*', '', meta_text)
        meta_text = re.sub(r'\s*```$', '', meta_text).strip()
        try:
            ticket = json.loads(meta_text)
            ticket["escalation"] = str(ticket.get("escalation", False)).lower() == "true"
            ticket["parse_error"] = False
            return reply, ticket
        except json.JSONDecodeError:
            # Greedy match — .*? stops at first } which breaks multiline JSON
            json_match = re.search(r'\{.*\}', meta_text, re.DOTALL)
            if json_match:
                try:
                    ticket = json.loads(json_match.group())
                    ticket["escalation"] = str(ticket.get("escalation", False)).lower() == "true"
                    ticket["parse_error"] = False
                    return reply, ticket
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"  [WARNING] JSON parse failed in META block: {e}")
                        print(f"  [WARNING] META content was: {meta_text[:200]!r}")
    else:
        if verbose:
            print(f"  [WARNING] No [META] tag found in LLM output")

    if verbose:
        print(f"  [WARNING] Could not parse [META] block — defaulting ticket to in_progress")
    return reply, {
        "ticket_status": "in_progress",
        "escalation": False,
        "reasoning": "Could not parse META block — defaulted to in_progress",
        "parse_error": True
    }


# ── Core function ────────────────────────────────────────────────────────────

def generate_response(test_case: dict, config: dict, verbose: bool = True) -> dict:
    """
    Run one test case through the CRM auto-responder pipeline.

    Args:
        test_case : One case from data_loader (joined from emails.json + context.json + ground_truth.json)
        config    : Full config dict from load_config()
        verbose   : When False, suppresses all progress and warning prints.
                    playground.py passes True; pytest passes False via run_case().

    Returns:
        dict with keys:
            id                  : test case ID (e.g. "TC001")
            input_email         : original email body
            retrieved_context   : context chunks used (pass-through for evaluators)
            generated_reply     : LLM-generated CRM reply
            predicted_ticket_status : "open"|"in_progress"|"escalated"|"resolved"
            predicted_escalation    : bool
            ticket_reasoning    : why the LLM chose this status
            model_used          : which Ollama model generated the reply
            ground_truth        : original ground truth (pass-through for evaluators)
    """
    sut_cfg  = get_sut_config(config)
    provider = sut_cfg["provider"]
    model    = sut_cfg["model"]

    rag_cfg  = get_rag_config(config)
    rag_mode = rag_cfg.get("mode", "simulated")

    if rag_mode == "live":
        from src.rag.retriever import retrieve
        context = retrieve(test_case["email_body"], config)
        if verbose:
            print(f"  → RAG mode: live — retrieved {len(context)} chunks from ChromaDB")
    else:
        context = test_case["retrieved_chunks"]

    if verbose:
        print(f"\n  → Generating reply + ticket prediction with {provider}/{model} (single call)...")

    # ONE call — reply and ticket metadata together
    llm    = _get_llm(sut_cfg)
    prompt = _build_combined_prompt(
        test_case["email_subject"], test_case["email_body"], context
    )
    # Retry up to 3 times on transient rate limit errors (Cerebras queue_exceeded, Groq 429)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            raw_output = llm.invoke(prompt).content.strip()
            break
        except RateLimitError as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"SUT rate limit hit after {max_retries} retries — {type(e).__name__}: {str(e)[:120]}"
                ) from e
            wait = 15 * attempt  # 15s, 30s, 45s
            if verbose:
                print(f"  [rate limit] SUT 429 on attempt {attempt}/{max_retries} — retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            if provider == "ollama" and ("Connection refused" in str(e) or "ConnectError" in type(e).__name__):
                raise RuntimeError(
                    f"Cannot connect to Ollama at {sut_cfg['base_url']}\n"
                    f"  Make sure Ollama is running: ollama serve\n"
                    f"  Then check the model is pulled: ollama pull {model}"
                ) from e
            raise
    generated_reply, ticket_result = _parse_combined_output(raw_output, verbose=verbose)

    # Validate ticket_status is one of the 4 allowed values
    valid_statuses = {"open", "in_progress", "escalated", "resolved"}
    ticket_status  = ticket_result.get("ticket_status", "in_progress")
    if ticket_status not in valid_statuses:
        if verbose:
            print(f"  [WARNING] Invalid ticket_status '{ticket_status}' — defaulting to in_progress")
        ticket_result["ticket_status"] = "in_progress"

    # Reconstruct ground_truth sub-dict for downstream evaluators
    ground_truth = {
        "expected_reply"         : test_case["expected_reply"],
        "expected_ticket_status" : test_case["expected_ticket_status"],
        "expected_escalation"    : test_case["expected_escalation"],
        "expected_tone"          : test_case["expected_tone"],
        "key_facts_to_include"   : test_case["key_facts_to_include"],
    }

    return {
        "id"                      : test_case["id"],
        "category"                : test_case["category"],
        "intent"                  : test_case["intent"],
        "input_email"             : test_case["email_body"],
        "email_subject"           : test_case["email_subject"],
        "customer_id"             : test_case["customer_id"],
        "ticket_id"               : test_case["ticket_id"],
        "retrieved_context"       : context,
        "generated_reply"         : generated_reply,
        "predicted_ticket_status" : ticket_result.get("ticket_status", "in_progress"),
        "predicted_escalation"    : ticket_result.get("escalation", False),
        "ticket_reasoning"        : ticket_result.get("reasoning", ""),
        "meta_parse_error"        : ticket_result.get("parse_error", False),
        "model_used"              : model,
        "ground_truth"            : ground_truth,
        "validation_focus"        : test_case.get("validation_focus", []),
        "llm_syndrome_watch"      : test_case.get("llm_syndrome_watch", ""),
    }


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich import print as rprint
    from src.utils.data_loader import get_case_by_id

    console = Console()
    config  = load_config()

    # Load TC001 via data_loader (joins emails + context + ground_truth)
    test_case = get_case_by_id("TC001")

    console.print(f"\n[bold cyan]╔══ CRM RESPONDER — SMOKE TEST (TC001) ══╗[/bold cyan]")
    console.print(f"[bold]Case:[/bold]     {test_case['id']} — {test_case['intent']}")
    console.print(f"[bold]Email:[/bold]    {test_case['email_subject']}")
    console.print(f"[bold]Provider:[/bold] {config['llm'].get('sut_provider', 'ollama')} / {config['llm']['sut_model']}\n")

    result = generate_response(test_case, config)

    console.print(Panel(
        result["generated_reply"],
        title="[green]Generated Reply[/green]",
        border_style="green"
    ))

    rprint(f"\n[bold]Predicted ticket status:[/bold] [yellow]{result['predicted_ticket_status']}[/yellow]")
    rprint(f"[bold]Predicted escalation  :[/bold] [yellow]{result['predicted_escalation']}[/yellow]")
    rprint(f"[bold]Reasoning             :[/bold] {result['ticket_reasoning']}")

    rprint(f"\n[bold]Expected ticket status:[/bold] [cyan]{result['ground_truth']['expected_ticket_status']}[/cyan]")
    rprint(f"[bold]Expected escalation   :[/bold] [cyan]{result['ground_truth']['expected_escalation']}[/cyan]")

    match_status = result["predicted_ticket_status"] == result["ground_truth"]["expected_ticket_status"]
    match_escl   = result["predicted_escalation"]    == result["ground_truth"]["expected_escalation"]

    rprint(f"\n[bold]Ticket status match:[/bold] {'[green]✓ PASS[/green]' if match_status else '[red]✗ FAIL[/red]'}")
    rprint(f"[bold]Escalation match   :[/bold] {'[green]✓ PASS[/green]' if match_escl   else '[red]✗ FAIL[/red]'}")

    console.print(f"\n[bold green]✓ CRM Responder working[/bold green]\n")
