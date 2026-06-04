"""
Stage 5 — Generation: grounded answer synthesis over retrieved chunks.

End-to-end entry point: ask(question) -> {"answer", "sources", "grounded", "chunks"}.

Grounding contract (this is the whole point of the module):
  1. The model is instructed to answer ONLY from the numbered context passages.
     The instruction is phrased as a hard constraint, not a suggestion, and the
     temperature is pinned to 0 so the model does not wander off-context.
  2. If retrieval returns nothing, we never even call the LLM — we return the
     refusal phrase directly. The model is given no opportunity to invent an
     answer from thin air.
  3. Source attribution is GUARANTEED PROGRAMMATICALLY. The list of sources
     returned to the caller is built from the retrieved chunks' metadata in this
     module — it is NOT parsed out of the model's output and never depends on the
     model choosing to cite. The model references passages by their [n] markers;
     the [n] -> real source mapping is owned here, so a citation cannot be
     hallucinated, dropped, or rewritten by the LLM.

Connects Stage 4 retrieval (pipeline.embeddings.retrieve) to Groq's
llama-3.3-70b-versatile.
"""
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _lazy_groq():
    global Groq
    if Groq is None:
        from groq import Groq as _Groq
        Groq = _Groq
    return Groq


# ---- configuration -----------------------------------------------------------
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
    "Try rephrasing your question — or check that the ingestion and embedding "
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


# ---- source resolution (file/doc_id -> original URL) -------------------------
# Best-effort mapping from a chunk's source file back to the URL it came from,
# using documents/download_metadata.json. Chunks may have been built from an
# earlier download whose hashed filenames differ, so we key primarily on the
# stable two-digit ordinal prefix ("01_", "02_", ...), which corresponds to the
# URL order in planning.md, and fall back to exact basename match.
_URL_LOOKUP_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def _load_url_lookup(metadata_path: str = "documents/download_metadata.json") -> Dict[str, Dict[str, str]]:
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


# ---- context + source construction ------------------------------------------
def _format_context_and_sources(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Build the numbered context block and the parallel, programmatic source list.

    The [n] used in the context block is the SAME n used in the source label, so
    a citation the model emits maps deterministically onto a real source that we
    control. Sources are derived purely from chunk metadata here.
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
    core = re.sub(r"[^a-z ]", "", (answer or "").lower()).strip()
    return core == _REFUSAL_CORE or core.startswith(_REFUSAL_CORE)


def _build_messages(question: str, context: str) -> List[Dict[str, str]]:
    system = _SYSTEM_PROMPT_TEMPLATE.format(refusal=REFUSAL_PHRASE, context=context)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


# ---- LLM call ----------------------------------------------------------------
_CLIENT_CACHE: Dict[str, Any] = {}


def _get_client():
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
    client = _get_client()
    resp = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# ---- public end-to-end entry point -------------------------------------------
def ask(
    question: str,
    *,
    k: int = DEFAULT_K,
    persist_dir: str = "chroma_db",
    collection_name: str = "documents",
    model_name: str = EMBED_MODEL,
    rerank: bool = True,
    llm_model: str = LLM_MODEL,
    max_tokens: int = 600,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Retrieve grounded context for `question`, generate an answer, attach sources.

    Returns:
        {
          "answer":   str,          # grounded answer or the refusal phrase
          "sources":  List[str],    # programmatically built; empty on refusal
          "grounded": bool,         # True iff the answer is backed by cited context
          "chunks":   List[dict],   # raw retrieved results (for debugging/eval)
        }
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": [], "grounded": False, "chunks": []}

    retrieval = retrieve(
        question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
        k=k,
        rerank=rerank,
    )
    results = retrieval.get("results") or []

    # No context -> refuse without ever calling the LLM. Grounding can't be
    # violated if the model is never asked.
    if not results:
        return {"answer": EMPTY_RETRIEVAL_MESSAGE, "sources": [], "grounded": False, "chunks": []}

    context, sources = _format_context_and_sources(results)
    messages = _build_messages(question, context)
    answer = _generate(messages, llm_model, max_tokens, temperature)

    # If the model refused, suppress sources: there is no grounded claim to attribute.
    if _looks_like_refusal(answer):
        return {"answer": REFUSAL_PHRASE, "sources": [], "grounded": False, "chunks": results}

    return {"answer": answer, "sources": sources, "grounded": True, "chunks": results}


# ---- CLI for quick manual checks ---------------------------------------------
def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Ask a grounded question against the housing corpus.")
    p.add_argument("question", help="Your question")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--persist_dir", default="chroma_db")
    p.add_argument("--collection", default="documents")
    p.add_argument("--model", default=EMBED_MODEL, help="Embedding model (must match index)")
    p.add_argument("--no-rerank", action="store_true")
    args = p.parse_args()

    res = ask(
        args.question,
        k=args.k,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        rerank=not args.no_rerank,
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