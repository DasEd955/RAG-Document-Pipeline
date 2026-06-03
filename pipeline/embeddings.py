"""
Embedding + retrieval over ChromaDB using Sentence-Transformers.

Public API (positional call order preserved so existing CLIs keep working):
    embed_and_index(chunks_jsonl, persist_dir, collection_name, model_name, batch_size, overwrite)
    retrieve(query, persist_dir, collection_name, model_name, k, rerank=True)

Key behaviours vs. the old version:
- The cross-encoder reranker actually runs now (CrossEncoder was leaking to a
  function-local before, so the rerank branch was dead code).
- The collection uses cosine space, so the returned distance IS (1 - cosine).
  No query-time re-encoding of documents.
- Models are loaded once per process and cached.
- The embedding model name is recorded in collection metadata and verified at
  query time, so an index/query model mismatch fails loudly.
- No silent in-memory fallback for the Chroma client.

IMPORTANT: this version creates the collection with cosine space. If you have an
existing index built by the old (L2) code, re-run embedding with --overwrite
once, or the cosine numbers from `retrieve` will be wrong.
"""
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import json

# ---- optional / lazy imports -------------------------------------------------
SentenceTransformer = None
CrossEncoder = None
chromadb = None
np = None


def _lazy_imports():
    # Every name assigned here MUST be declared global. The original bug was
    # omitting CrossEncoder from this list, which made the import a discarded
    # local and silently disabled all reranking.
    global SentenceTransformer, CrossEncoder, chromadb, np
    try:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST
    except Exception:
        SentenceTransformer = None
    try:
        from sentence_transformers import CrossEncoder as _CE
        CrossEncoder = _CE
    except Exception:
        CrossEncoder = None
    try:
        import chromadb as _chromadb
        chromadb = _chromadb
    except Exception:
        chromadb = None
    try:
        import numpy as _np
        np = _np
    except Exception:
        np = None


_lazy_imports()

# ---- model caching (load each model once per process) ------------------------
_MODEL_CACHE: Dict[str, Any] = {}
_RERANKER_CACHE: Dict[str, Any] = {}
_DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed: pip install sentence-transformers")
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _get_reranker(name: str = _DEFAULT_RERANKER):
    """Return a cached CrossEncoder, or None if the dependency is unavailable."""
    if CrossEncoder is None:
        return None
    if name not in _RERANKER_CACHE:
        try:
            _RERANKER_CACHE[name] = CrossEncoder(name)
        except Exception:
            _RERANKER_CACHE[name] = None
    return _RERANKER_CACHE[name]


def _normalize(emb):
    """Unit-normalize a 2D embedding array. No-op if numpy is missing."""
    if np is None:
        return emb
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms


# ---- io ----------------------------------------------------------------------
def load_chunks(jsonl_path: str) -> Iterable[Dict[str, Any]]:
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"Chunks file not found: {jsonl_path}")
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def ensure_client(persist_dir: str = "chroma_db"):
    """Return a persistent Chroma client.

    Deliberately does NOT fall back to an in-memory client: a silent ephemeral
    store looks like it works but loses all data between runs.
    """
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Install with `pip install chromadb`")
    return chromadb.PersistentClient(path=persist_dir)


# ---- indexing ----------------------------------------------------------------
def embed_and_index(
    chunks_jsonl: str = "chunks.jsonl",
    persist_dir: str = "chroma_db",
    collection_name: str = "documents",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
    overwrite: bool = False,
) -> Any:
    """Embed chunks and index them into a persistent, cosine-space Chroma collection."""
    model = _get_model(model_name)
    client = ensure_client(persist_dir)

    coll_metadata = {"hnsw:space": "cosine", "embedding_model": model_name}

    if overwrite:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    try:
        collection = client.get_or_create_collection(name=collection_name, metadata=coll_metadata)
    except Exception:
        collection = client.create_collection(name=collection_name, metadata=coll_metadata)

    # Guard: never mix models within one collection.
    existing_model = (collection.metadata or {}).get("embedding_model")
    if existing_model and existing_model != model_name:
        raise RuntimeError(
            f"Collection '{collection_name}' was built with '{existing_model}', "
            f"but you are indexing with '{model_name}'. Re-run with overwrite=True "
            f"(or --overwrite) to rebuild, or use the original model."
        )

    ids_batch: List[str] = []
    docs_batch: List[str] = []
    met_batch: List[Dict[str, Any]] = []

    def flush():
        if not ids_batch:
            return
        emb = model.encode(docs_batch, convert_to_numpy=True, show_progress_bar=False)
        emb = _normalize(emb)
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        collection.add(ids=ids_batch, documents=docs_batch, metadatas=met_batch, embeddings=emb_list)
        ids_batch.clear()
        docs_batch.clear()
        met_batch.clear()

    for item in load_chunks(chunks_jsonl):
        text = item.get("text") or ""
        doc_id = item.get("doc_id") or ""
        chunk_index = item.get("chunk_index")
        ids_batch.append(f"{doc_id}__{chunk_index}")
        docs_batch.append(text)
        met_batch.append({
            "doc_id": doc_id,
            "source": item.get("source"),
            "chunk_index": chunk_index,
            # NOTE: char_span is a list. Older Chroma tolerates list metadata;
            # newer versions require scalars. If you upgrade Chroma and this
            # rejects, swap to json.dumps(item.get("char_span")).
            "char_span": item.get("char_span"),
            "token_count": item.get("token_count"),
        })
        if len(ids_batch) >= batch_size:
            flush()
    flush()
    return collection


# ---- retrieval ---------------------------------------------------------------
def retrieve(
    query: str,
    persist_dir: str = "chroma_db",
    collection_name: str = "documents",
    model_name: str = "all-MiniLM-L6-v2",
    k: int = 3,
    rerank: bool = True,
) -> Dict[str, Any]:
    """Return the top-k chunks for `query`.

    Pulls a wider candidate set from the vector store, then (if a cross-encoder
    is available) reranks by true query-document relevance and keeps the top k.
    """
    model = _get_model(model_name)
    client = ensure_client(persist_dir)
    collection = client.get_collection(name=collection_name)

    # Guard: query model must match the model the index was built with.
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with and built_with != model_name:
        raise RuntimeError(
            f"Index '{collection_name}' was built with '{built_with}', "
            f"but you queried with '{model_name}'. Use the same model for both."
        )

    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q_emb = _normalize(q_emb)
    q_list = q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb)

    candidate_n = max(k, 20)
    res = collection.query(
        query_embeddings=q_list,
        n_results=candidate_n,
        include=["metadatas", "documents", "distances"],
    )

    docs_list = (res.get("documents") or [[]])[0]
    metas_list = (res.get("metadatas") or [[]])[0]
    dists_list = (res.get("distances") or [[]])[0]
    ids_list = (res.get("ids") or [[]])[0]

    results: List[Dict[str, Any]] = []
    for i in range(len(docs_list)):
        dist = dists_list[i] if i < len(dists_list) else None
        # Collection is cosine space: distance == 1 - cosine_similarity.
        cos = (1.0 - dist) if dist is not None else None
        results.append({
            "id": ids_list[i] if i < len(ids_list) else None,
            "document": docs_list[i],
            "metadata": metas_list[i] if i < len(metas_list) else None,
            "distance": dist,
            "cosine_similarity": cos,
        })

    reranker = _get_reranker() if rerank else None
    if reranker is not None and results:
        pairs = [[query, r["document"] or ""] for r in results]
        scores = reranker.predict(pairs)
        scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda r: r.get("rerank_score", float("-inf")), reverse=True)
    else:
        results.sort(
            key=lambda r: (r.get("cosine_similarity") if r.get("cosine_similarity") is not None else -1.0),
            reverse=True,
        )

    return {"query": query, "results": results[:k], "reranked": reranker is not None}