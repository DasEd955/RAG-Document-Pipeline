"""
Query.py - Generation Stage; grounded answer synthesis over retrieved chunks.

End-to-end entry point: ask(question) -> {"answer", "sources", "grounded", "chunks"}.

Grounding contract (this is the whole point of the module):
  1. The model is instructed to answer ONLY from the numbered context passages.
     The instruction is phrased as a hard constraint, not a suggestion, and the
     temperature is pinned to 0 so the model does not wander off-context.
  2. If retrieval returns nothing, we never even call the LLM; we return the
     refusal phrase directly. The model is given no opportunity to invent an
     answer from thin air.
  3. Source attribution is GUARANTEED PROGRAMMATICALLY. The list of sources
     returned to the caller is built from the retrieved chunks' metadata in this
     module; it is NOT parsed out of the model's output and never depends on the
     model choosing to cite. The model references passages by their [n] markers;
     the [n] -> real source mapping is owned here, so a citation cannot be
     hallucinated, dropped, or rewritten by the LLM.

Connects Stage 4 retrieval (pipeline.embeddings.retrieve) to Groq's
llama-3.3-70b-versatile.
"""
import os
import re
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root (parent of app/) is importable, so `pipeline` resolves
# whether this is run as `python app/query.py`, `python -m app.query`, or imported.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.embeddings import retrieve

# Load GROQ_API_KEY (and friends) from a local .env if python-dotenv is present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Groq is imported lazily inside the call path so that retrieval-only / test
# usage does not hard-require the SDK or an API key.
Groq = None

def _lazy_groq() -> type:
    """Lazily import the Groq SDK and cache the class.

    Returns the Groq class without instantiating a client. Used to defer
    dependency loading until the first call that requires it.

    Returns:
        type: The Groq client class.

    Raises:
        ImportError: If groq is not installed.
    """
    global Groq
    if Groq is None:
        from groq import Groq as _Groq
        Groq = _Groq
    return Groq


# ---- Configuration -----------------------------------------------------------
LLM_MODEL = "llama-3.3-70b-versatile"          # Groq generation model
EMBED_MODEL = "all-mpnet-base-v2"              # MUST match the indexed model
DEFAULT_K = 5

# The exact phrase the model must emit when the context is insufficient, and the
# phrase we return ourselves when retrieval is empty. Kept in one place so the
# prompt, the refusal short-circuit, and the refusal detector never drift apart.
REFUSAL_PHRASE = (
    "I don't have enough information in the loaded documents to answer that."
)

EMPTY_RETRIEVAL_MESSAGE = (
    "I couldn't find anything relevant in the loaded housing documents. "
    "Try rephrasing your question, or check that the ingestion and embedding "
    "pipeline has been run."
)

_SYSTEM_PROMPT_TEMPLATE = """You are a grounded question-answering assistant for off-campus student housing near Penn State / State College, PA.

You will be given a user question and a set of numbered context passages retrieved from a fixed corpus of forum posts, reviews, news articles, and official guides. You must obey ALL of the following rules:

1. Answer using ONLY the information contained in the numbered context passages below. The passages are your single source of truth.
2. Do NOT use any outside or prior knowledge. Do NOT guess, extrapolate, or add any fact that is not explicitly supported by the passages.
3. If the passages do not contain enough information to answer the question, you MUST reply with this exact sentence and nothing else:
{refusal}
4. When you state a fact, cite the passage number(s) it came from using square-bracket markers, e.g. [1] or [2][3]. Only ever use the [n] markers that appear in the context — never invent source names, URLs, dates, or numbers.
5. Be concise and factual. Prefer the wording and sentiment actually expressed in the passages over your own phrasing.

Context passages:
{context}"""

# A model output is treated as a refusal if it is essentially just the refusal
# sentence (allowing for minor punctuation/whitespace differences).
_REFUSAL_CORE = re.sub(r"[^a-z ]", "", REFUSAL_PHRASE.lower()).strip()

# ---- Conversational Memory Prompts -------------------------------------------
# Used to rewrite a context-dependent follow-up into a standalone retrieval query.
_CONDENSE_SYSTEM_PROMPT = """You rewrite a user's latest message into a single standalone search query for a document retrieval system.

Given the conversation so far and the latest user message, produce one self-contained question that:
- resolves pronouns and references (e.g. "it", "there", "that place", "they") to the specific entity named earlier in the conversation,
- carries over any constraints still in effect (location, budget, building or landlord name),
- removes conversational filler.

Output ONLY the rewritten query text and nothing else. If the latest message is already self-contained, output it unchanged."""

# Inserted into the generation system prompt when history is present. It supplies
# prior turns for reference resolution while reasserting the grounding contract so
# the model never treats earlier turns as a source of new facts.
_HISTORY_INSTRUCTION = (
    "The earlier conversation turns below are provided ONLY to help you interpret "
    "references in the current question (for example, resolving \"it\" or \"there\" "
    "to a specific place). They are NOT a source of facts: you must still answer "
    "using ONLY the numbered context passages, and still refuse if those passages "
    "are insufficient.\n\n"
    "Conversation so far:\n{history}\n\n"
)


# ---- Source Resolution (file/doc_id -> original URL) -------------------------
# Best-effort mapping from a chunk's source file back to the URL it came from,
# using documents/download_metadata.json. Chunks may have been built from an
# earlier download whose hashed filenames differ, so we key primarily on the
# stable two-digit ordinal prefix ("01_", "02_", ...), which corresponds to the
# URL order in planning.md, and fall back to exact basename match.
_URL_LOOKUP_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def _load_url_lookup(metadata_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """Load and cache a mapping from file ordinals/basenames to original URLs.

    Parses documents/download_metadata.json (generated by download_documents.py)
    and builds lookups keyed by filename prefix ("01_", "02_", ...) and basename.
    Results are cached in _URL_LOOKUP_CACHE to avoid repeated file I/O.

    Args:
        metadata_path (str, optional): Path to download_metadata.json. Defaults to
                                       the project root's documents/download_metadata.json.

    Returns:
        Dict[str, Dict[str, str]]: A dict with keys:
            - by_prefix: Maps ordinal strings ("01", "02", ...) to URLs.
            - by_basename: Maps full filename to URL.
    """
    if metadata_path is None:
        metadata_path = os.path.join(_ROOT, "documents", "download_metadata.json")
    global _URL_LOOKUP_CACHE
    if _URL_LOOKUP_CACHE is not None:
        return _URL_LOOKUP_CACHE
    by_prefix: Dict[str, str] = {}
    by_basename: Dict[str, str] = {}
    try:
        data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        for item in data:
            url = item.get("url")
            file = item.get("file") or ""
            if not url or not file:
                continue
            base = re.split(r"[\\/]", file)[-1]
            by_basename[base] = url
            m = re.match(r"^(\d+)_", base)
            if m:
                by_prefix.setdefault(m.group(1).zfill(2), url)
    except Exception:
        pass
    _URL_LOOKUP_CACHE = {"by_prefix": by_prefix, "by_basename": by_basename}
    return _URL_LOOKUP_CACHE


def _resolve_url(source: str, doc_id: str) -> Optional[str]:
    """Resolve a chunk's source file or doc_id back to its original URL.

    Uses the ordinal prefix (e.g., "01_") from the filename or doc_id to look up
    the URL in the metadata table. Falls back to basename matching if the prefix
    lookup fails. Returns None if no match is found.

    Args:
        source (str): The file path from chunk metadata (e.g., "documents/01_....html").
        doc_id (str): The document ID from chunk metadata (e.g., "01_www.reddit.com...").

    Returns:
        Optional[str]: The original URL if found, None otherwise.

    Example:
        >>> _resolve_url("documents/01_www.reddit.com_...html", "01_www.reddit...")
        "https://www.reddit.com/r/PennStateUniversity/comments/..."
    """
    lookup = _load_url_lookup()
    base = re.split(r"[\\/]", source or "")[-1] if source else ""
    if base and base in lookup["by_basename"]:
        return lookup["by_basename"][base]
    for key in (base, doc_id or ""):
        m = re.match(r"^(\d+)_", key)
        if m:
            url = lookup["by_prefix"].get(m.group(1).zfill(2))
            if url:
                return url
    return None


# ---- Context + Source Construction ------------------------------------------
def _format_context_and_sources(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Build numbered context passages and their programmatic source attribution.

    Constructs a context block "[1] passage text", "[2] passage text", ... and a
    parallel source list ["[1] URL (sim X.XX)", "[2] URL (sim Y.YY)", ...]. The
    indices are guaranteed to align, so any [n] citation in the model output maps
    to the correct source. Sources are derived solely from chunk metadata; they
    are never parsed from model output.

    Args:
        results (List[Dict[str, Any]]): Retrieved chunk results from retrieve(),
                                        each with keys: metadata, document,
                                        cosine_similarity.

    Returns:
        Tuple[str, List[str]]: A 2-tuple:
            - context (str): Newline-separated numbered passages "[1] ...", "[2] ...", etc.
            - sources (List[str]): Newline-separated source attributions with [n] markers.

    Example:
        >>> ctx, srcs = _format_context_and_sources(results)
        >>> print(ctx)
        [1] Passage about housing...
        <BLANKLINE>
        [2] Another passage...
    """
    blocks: List[str] = []
    sources: List[str] = []
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata") or {}
        text = (r.get("document") or "").strip()
        blocks.append(f"[{i}] {text}")

        doc_id = str(meta.get("doc_id") or "")
        source = str(meta.get("source") or "")
        url = _resolve_url(source, doc_id)
        # Prefer the original URL; otherwise fall back to the doc_id / file path.
        where = url or doc_id or source or "unknown source"

        cos = r.get("cosine_similarity")
        score = f" (similarity {cos:.2f})" if isinstance(cos, (int, float)) else ""
        sources.append(f"[{i}] {where}{score}")

    return "\n\n".join(blocks), sources


def _looks_like_refusal(answer: str) -> bool:
    """Detect whether a model output is a refusal (insufficient context) response.

    Normalizes the answer by lowercasing and stripping punctuation, then checks
    if it matches or starts with the canonical refusal phrase (minus punctuation).

    Args:
        answer (str): The model's response text.

    Returns:
        bool: True if answer is a refusal phrase, False otherwise.

    Example:
        >>> _looks_like_refusal("I don't have enough information in the loaded documents.")
        True
    """
    core = re.sub(r"[^a-z ]", "", (answer or "").lower()).strip()
    return core == _REFUSAL_CORE or core.startswith(_REFUSAL_CORE)


def _render_history(history: Optional[List[Dict[str, str]]], max_turns: int = 6) -> str:
    """Render the most recent conversation turns into a compact text transcript.

    Keeps only the last `max_turns` turns to bound prompt size, and formats each as
    "User: ..." / "Assistant: ..." lines. Used both to give the generator reference
    context and to feed the query condenser.

    Args:
        history (Optional[List[Dict[str, str]]]): Prior turns, each a dict with
            "question" and "answer" keys. None or empty yields "".
        max_turns (int, optional): Maximum number of trailing turns to include.
            Defaults to 6.

    Returns:
        str: A newline-joined transcript, or "" if there is no history.
    """
    if not history:
        return ""
    lines: List[str] = []
    for turn in history[-max_turns:]:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q:
            lines.append(f"User: {q}")
        if a:
            lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def condense_question(question: str, history: Optional[List[Dict[str, str]]],
                      llm_model: str = LLM_MODEL, max_turns: int = 6) -> str:
    """Rewrite a follow-up question into a standalone retrieval query using history.

    A follow-up like "Is it expensive?" embeds poorly because the entity ("it")
    lives in an earlier turn. This uses the LLM to fold the conversation into a
    single self-contained query (resolving pronouns and carrying over constraints)
    so semantic + BM25 retrieval see the full intent. With no history, or if the
    rewrite fails or comes back empty, the original question is returned unchanged.

    Args:
        question (str): The latest user message.
        history (Optional[List[Dict[str, str]]]): Prior turns (question/answer dicts).
        llm_model (str, optional): Groq model identifier. Defaults to LLM_MODEL.
        max_turns (int, optional): Trailing turns to consider. Defaults to 6.

    Returns:
        str: A standalone search query (or the original question if no rewrite is
             needed or possible).

    Example:
        >>> hist = [{"question": "Tell me about The Maxxen", "answer": "It is ..."}]
        >>> condense_question("Is it expensive?", hist)  # doctest: +SKIP
        'Is The Maxxen expensive?'
    """
    if not history:
        return question
    transcript = _render_history(history, max_turns)
    messages = [
        {"role": "system", "content": _CONDENSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation so far:\n{transcript}\n\n"
                                    f"Latest user message: {question}\n\nStandalone query:"},
    ]
    try:
        rewritten = _generate(messages, llm_model, max_tokens=128, temperature=0.0)
    except Exception:
        return question
    # Strip surrounding quotes/whitespace the model sometimes adds.
    rewritten = rewritten.strip().strip('"').strip()
    return rewritten or question


def _build_messages(question: str, context: str,
                    history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """Build a system + user message pair for the LLM.

    Formats the system prompt with the refusal phrase and numbered context, then
    creates the user message containing the question. When conversation history is
    supplied, a transcript is inserted into the system prompt for reference
    resolution only; the grounding rule (answer ONLY from the passages) is
    explicitly preserved, so prior turns never become a source of new facts.

    Args:
        question (str): The user's question.
        context (str): The numbered context block "[1] ...", "[2] ...", etc.
        history (Optional[List[Dict[str, str]]]): Prior turns for multi-turn
            context. Defaults to None (single-turn; prompt is unchanged).

    Returns:
        List[Dict[str, str]]: A list of message dicts ready for the LLM API:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    system = _SYSTEM_PROMPT_TEMPLATE.format(refusal=REFUSAL_PHRASE, context=context)
    transcript = _render_history(history)
    if transcript:
        block = _HISTORY_INSTRUCTION.format(history=transcript)
        # Insert the history block immediately before the context so the grounding
        # rules and the passages still bracket the model's attention.
        system = system.replace("Context passages:", block + "Context passages:")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


# ---- LLM Call ----------------------------------------------------------------

# Cached Groq client instance. Initialized on first use by _get_client().
_CLIENT_CACHE: Dict[str, Any] = {}


def _get_client() -> Any:
    """Get or create a cached Groq API client.

    Initializes the client once and caches it in _CLIENT_CACHE. Reads GROQ_API_KEY
    from the environment (or .env file if python-dotenv was used).

    Returns:
        Any: A Groq client instance.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set in the environment.
    """
    if "client" not in _CLIENT_CACHE:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or a .env file."
            )
        _CLIENT_CACHE["client"] = _lazy_groq()(api_key=api_key)
    return _CLIENT_CACHE["client"]


def _generate(messages: List[Dict[str, str]], llm_model: str, max_tokens: int,
              temperature: float) -> str:
    """Call the LLM to generate an answer from system + user messages.

    Uses the cached Groq client to make a completion request with the given
    configuration. Returns only the text of the first choice.

    Args:
        messages (List[Dict[str, str]]): Message list from _build_messages().
        llm_model (str): Groq model identifier (e.g., "llama-3.3-70b-versatile").
        max_tokens (int): Maximum tokens in the response.
        temperature (float): Sampling temperature (0 for deterministic, 1+ for random).

    Returns:
        str: The generated text response, trimmed of leading/trailing whitespace.

    Raises:
        Any Groq API errors (network, rate limit, etc.).
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# ---- Public End-to-End Entry Point -------------------------------------------
def ask(question: str,
        *,
        k: int = DEFAULT_K,
        persist_dir: str = "chroma_db",
        collection_name: str = "documents",
        model_name: str = EMBED_MODEL,
        rerank: bool = True,
        hybrid: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        source_boost: Optional[Dict[str, float]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        retrieval_query: Optional[str] = None,
        llm_model: str = LLM_MODEL,
        max_tokens: int = 600,
        temperature: float = 0.0,) -> Dict[str, Any]:
    """Retrieve grounded context for a question and generate a cited answer.

    End-to-end entry point: retrieves relevant chunks from the vector index via
    hybrid semantic + keyword search (with optional metadata filtering and source
    boosting), generates an answer grounded only in that context (with no
    hallucination), and attaches programmatic source attribution derived from chunk
    metadata. If retrieval is empty, refuses without calling the LLM for an answer.
    If the model refuses, suppresses sources (no grounded claim to attribute).

    Conversational memory: when `history` is supplied, a context-dependent
    follow-up is first condensed into a standalone retrieval query (so it embeds
    well), and the prior turns are added to the generation prompt for reference
    resolution only. The grounding contract is unchanged: the answer still comes
    solely from the retrieved passages.

    Args:
        question (str): The user's question.
        k (int, optional): Number of retrieval results to pass to the LLM.
                          Defaults to DEFAULT_K (5).
        persist_dir (str, optional): ChromaDB persistence dir. Defaults to "chroma_db".
        collection_name (str, optional): Collection name. Defaults to "documents".
        model_name (str, optional): Embedding model (must match index). Defaults to
                                    EMBED_MODEL ("all-mpnet-base-v2").
        rerank (bool, optional): Use cross-encoder reranking. Defaults to True.
        hybrid (bool, optional): Fuse BM25 keyword recall with semantic recall.
                                 Defaults to True.
        filters (dict, optional): Metadata filter spec forwarded to retrieve()
            (keys: source_contains, equals, date_after, date_before, min_rating).
            Defaults to None.
        source_boost (dict, optional): Map of source substring -> score multiplier
            forwarded to retrieve(). Defaults to None.
        history (list, optional): Prior conversation turns, each a dict with
            "question" and "answer" keys. Enables multi-turn context. Defaults to None.
        retrieval_query (str, optional): Pre-computed standalone query to retrieve
            with, overriding auto-condensation. Defaults to None.
        llm_model (str, optional): Groq model identifier. Defaults to LLM_MODEL
                                   ("llama-3.3-70b-versatile").
        max_tokens (int, optional): Max tokens in LLM response. Defaults to 600.
        temperature (float, optional): LLM sampling temperature. Defaults to 0.0
                                       (deterministic, enforces grounding).

    Returns:
        Dict[str, Any]: A result dict with keys:
            - answer (str): Grounded answer or the refusal phrase.
            - sources (List[str]): Programmatically built source attributions,
                                   empty if answer is a refusal.
            - grounded (bool): True iff the answer is backed by retrieved context.
            - chunks (List[dict]): Raw retrieval results (for debugging/eval).
            - retrieval_query (str): The query actually used for retrieval (the
                                     condensed form for follow-ups).

    Example:
        >>> res = ask("Is downtown State College expensive?", k=5)
        >>> print(res["answer"])
        Yes, downtown State College is expensive...
        >>> for src in res["sources"]:
        ...     print(src)
        [1] https://www.psucollegian.com/... (similarity 0.59)
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": [], "grounded": False,
                "chunks": [], "retrieval_query": question}

    # Conversational memory: fold prior turns into a standalone retrieval query so
    # a follow-up like "Is it expensive?" carries the entity from earlier turns.
    search_query = retrieval_query or condense_question(question, history, llm_model=llm_model)

    retrieval = retrieve(
        search_query,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
        k=k,
        rerank=rerank,
        hybrid=hybrid,
        filters=filters,
        source_boost=source_boost,
    )
    results = retrieval.get("results") or []

    # No Context -> Refuse without ever calling the LLM for an answer. Grounding
    # can't be violated if the model is never asked to produce one.
    if not results:
        return {"answer": EMPTY_RETRIEVAL_MESSAGE, "sources": [], "grounded": False,
                "chunks": [], "retrieval_query": search_query}

    context, sources = _format_context_and_sources(results)
    messages = _build_messages(question, context, history=history)
    answer = _generate(messages, llm_model, max_tokens, temperature)

    # If the model refused, suppress sources: there is no grounded claim to attribute.
    if _looks_like_refusal(answer):
        return {"answer": REFUSAL_PHRASE, "sources": [], "grounded": False,
                "chunks": results, "retrieval_query": search_query}

    return {"answer": answer, "sources": sources, "grounded": True,
            "chunks": results, "retrieval_query": search_query}


# ---- CLI For Quick Manual Checks ---------------------------------------------
def _cli() -> None:
    """Command-line interface for asking grounded questions.

    Parses arguments for a question and optional retrieval/embedding parameters,
    calls ask(), and prints the answer and sources.
    """
    import argparse
    p = argparse.ArgumentParser(description="Ask a grounded question against the housing corpus.")
    p.add_argument("question", help="Your question")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--persist_dir", default="chroma_db")
    p.add_argument("--collection", default="documents")
    p.add_argument("--model", default=EMBED_MODEL, help="Embedding model (must match index)")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-hybrid", action="store_true", help="Disable BM25 keyword recall (semantic only)")
    p.add_argument("--source", default=None, help="Only use chunks whose source contains this substring")
    args = p.parse_args()

    filters = {"source_contains": args.source} if args.source else None

    res = ask(
        args.question,
        k=args.k,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        rerank=not args.no_rerank,
        hybrid=not args.no_hybrid,
        filters=filters,
    )
    print("\n=== Answer ===\n")
    print(res["answer"])
    print("\n=== Retrieved from ===\n")
    if res["sources"]:
        for s in res["sources"]:
            print(f"  {s}")
    else:
        print("  (no sources — answer is a refusal)")
    print()


if __name__ == "__main__":
    _cli()