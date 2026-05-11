"""
retriever.py
------------
Real RAG retrieval layer for the BSFI CRM Auto-Responder Evaluation Framework.

Two public functions:

    build_index(config)
        Reads all PDFs from data/policy_docs/, chunks them, embeds using
        sentence-transformers, and persists a ChromaDB collection to
        data/chroma_db/. Safe to re-run — wipes and rebuilds the collection.

    retrieve(query, config) -> list[str]
        Embeds the query and returns the top-k most relevant chunk strings
        from the ChromaDB collection. Called by crm_responder when
        rag.mode = "live".

Used by:
    src/pipeline/crm_responder.py  — retrieve() called per test case
    python -m src.rag.retriever    — standalone index build + smoke test

Standalone usage:
    python -m src.rag.retriever          # builds index from policy_docs/
    python -m src.rag.retriever --query "personal loan eligibility"
"""

import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from src.utils.config_loader import load_config, get_rag_config

COLLECTION_NAME = "bsfi_policy_docs"

_embedding_model_cache: SentenceTransformer | None = None


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    global _embedding_model_cache
    if _embedding_model_cache is None:
        _embedding_model_cache = SentenceTransformer(model_name)
    return _embedding_model_cache


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _extract_text_from_txt(txt_path: Path) -> str:
    """Read plain text file."""
    return txt_path.read_text(encoding="utf-8")


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    chunk_size and chunk_overlap are in approximate word counts,
    which correlates loosely with tokens for English text.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        # Strip leading/trailing whitespace and collapse internal whitespace
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def build_index(config: dict = None) -> int:
    """
    Build ChromaDB index from all PDFs in data/policy_docs/.

    Wipes any existing collection and rebuilds from scratch — safe to re-run
    whenever policy documents are updated.

    Args:
        config: Full config dict. Loads from config.yaml if not provided.

    Returns:
        Total number of chunks indexed.

    Raises:
        FileNotFoundError: If policy_docs/ directory does not exist or is empty.
    """
    config     = config or load_config()
    rag_cfg    = get_rag_config(config)
    model_name = rag_cfg["embedding_model"]
    chunk_size  = rag_cfg["chunk_size"]
    chunk_overlap = rag_cfg["chunk_overlap"]

    project_root = Path(__file__).resolve().parents[2]
    docs_path    = (project_root / rag_cfg["documents_path"]).resolve()
    chroma_path  = (project_root / rag_cfg["chroma_db_path"]).resolve()

    if not docs_path.exists():
        raise FileNotFoundError(f"Policy docs directory not found: {docs_path}")

    pdf_files = sorted(docs_path.glob("*.pdf")) + sorted(docs_path.glob("*.txt"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF or TXT files found in: {docs_path}")

    print(f"  [RAG] Loading embedding model: {model_name}")
    model = _get_embedding_model(model_name)

    chroma_path.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(chroma_path))

    # Wipe and recreate the collection so re-runs are idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_chunks   = []
    all_ids      = []
    all_metadata = []

    for pdf_path in pdf_files:
        print(f"  [RAG] Parsing: {pdf_path.name}")
        if pdf_path.suffix == ".txt":
            text = _extract_text_from_txt(pdf_path)
        else:
            text = _extract_text_from_pdf(pdf_path)
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        print(f"         → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{pdf_path.stem}_{i:04d}")
            all_metadata.append({"source": pdf_path.name, "chunk_index": i})

    print(f"  [RAG] Embedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # ChromaDB has a batch size limit — add in batches of 500
    batch_size = 500
    for start in range(0, len(all_chunks), batch_size):
        end = start + batch_size
        collection.add(
            documents=all_chunks[start:end],
            embeddings=embeddings[start:end],
            ids=all_ids[start:end],
            metadatas=all_metadata[start:end],
        )

    print(f"  [RAG] Index built: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
    print(f"  [RAG] Persisted to: {chroma_path}")
    return len(all_chunks)


def retrieve(query: str, config: dict = None) -> list[str]:
    """
    Retrieve the top-k most relevant policy chunks for a given query.

    Args:
        query  : The customer email body or query string.
        config : Full config dict. Loads from config.yaml if not provided.

    Returns:
        List of chunk strings (plain text), most relevant first.

    Raises:
        RuntimeError: If the ChromaDB index has not been built yet.
    """
    config    = config or load_config()
    rag_cfg   = get_rag_config(config)
    model_name = rag_cfg["embedding_model"]
    top_k     = rag_cfg["top_k"]

    project_root = Path(__file__).resolve().parents[2]
    chroma_path  = (project_root / rag_cfg["chroma_db_path"]).resolve()

    if not chroma_path.exists():
        raise RuntimeError(
            f"ChromaDB index not found at {chroma_path}\n"
            f"  Build it first: python -m src.rag.retriever"
        )

    client     = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found in ChromaDB at {chroma_path}\n"
            f"  Build it first: python -m src.rag.retriever"
        )

    model      = _get_embedding_model(model_name)
    query_emb  = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0] if results["documents"] else []
    return chunks


# ── Standalone: build index + optional smoke test ────────────────────────────

if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.panel import Panel
    from rich import print as rprint

    console = Console()
    config  = load_config()

    console.print("\n[bold cyan]╔══ RAG RETRIEVER ══╗[/bold cyan]\n")

    # Build index
    console.print("[bold yellow]── Building index from policy_docs/ ──[/bold yellow]")
    total = build_index(config)
    console.print(f"\n[bold green]✓ Index built: {total} total chunks[/bold green]\n")

    # Optional query smoke test
    query_arg = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--query" and i + 1 < len(sys.argv) - 1:
            query_arg = sys.argv[i + 2]
            break

    if query_arg:
        console.print(f"[bold yellow]── Smoke test query ──[/bold yellow]")
        console.print(f"Query: [italic]{query_arg}[/italic]\n")
        chunks = retrieve(query_arg, config)
        for i, chunk in enumerate(chunks, 1):
            console.print(Panel(chunk, title=f"[green]Chunk {i}[/green]", border_style="green"))
    else:
        # Default smoke test queries
        test_queries = [
            "personal loan eligibility requirements",
            "EMI payment failed what to do",
            "interest rate dispute escalation",
            "investment advice stocks",
        ]
        console.print("[bold yellow]── Default smoke test queries ──[/bold yellow]")
        for query in test_queries:
            console.print(f"\n[bold]Query:[/bold] [italic]{query}[/italic]")
            chunks = retrieve(query, config)
            console.print(f"  → {len(chunks)} chunks retrieved")
            if chunks:
                console.print(f"  → Top chunk: {chunks[0][:120]}...")

    console.print("\n[bold green]✓ Retriever working[/bold green]\n")
