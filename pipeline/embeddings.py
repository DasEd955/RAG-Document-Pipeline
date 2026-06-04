"""
Embeddings.py - Embedding + retrieval over ChromaDB using Sentence-Transformers.

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

# ---- Optional / Lazy Imports -------------------------------------------------
SentenceTransformer = None
CrossEncoder = None
chromadb = None
np = None


def _lazy_imports() -> None:
    """Import optional dependencies at runtime, setting global module handles.

    Imports SentenceTransformer, CrossEncoder, chromadb, and numpy. Each import
    is wrapped in a try-except to gracefully degrade if dependencies are missing.
    Every global must be declared before assignment to avoid local shadowing.
    """
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

# ---- Model Caching (Load Each Model Once per Process) ------------------------
_MODEL_CACHE: Dict[str, Any] = {}
_RERANKER_CACHE: Dict[str, Any] = {}
_DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model(model_name: str) -> Any:
    """Load or retrieve a cached embedding model by name.

    Models are cached in _MODEL_CACHE so repeated calls return the same instance
    without reloading.

    Args:
        model_name (str): Hugging Face model identifier (e.g., "all-mpnet-base-v2").

    Returns:
        Any: A SentenceTransformer model instance.

    Raises:
        RuntimeError: If sentence-transformers is not installed.
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed: pip install sentence-transformers")
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _get_reranker(name: str = _DEFAULT_RERANKER) -> Optional[Any]:
    """Load or retrieve a cached cross-encoder reranking model.

    The cross-encoder jointly scores (query, document) pairs to refine retrieval
    results. If CrossEncoder is unavailable, returns None and reranking is skipped.

    Args:
        name (str, optional): Cross-encoder model identifier. Defaults to
                             "cross-encoder/ms-marco-MiniLM-L-6-v2".

    Returns:
        Optional[Any]: A CrossEncoder instance, or None if unavailable or load fails.
    """
    if CrossEncoder is None:
        return None
    if name not in _RERANKER_CACHE:
        try:
            _RERANKER_CACHE[name] = CrossEncoder(name)
        except Exception:
            _RERANKER_CACHE[name] = None
    return _RERANKER_CACHE[name]


def _normalize(emb: Any) -> Any:
    """Unit-normalize a 2D embedding array to unit length for cosine distance.

    If numpy is unavailable, returns the input unchanged (graceful degradation).
    Normalization is required for cosine similarity in ChromaDB.

    Args:
        emb (Any): A 2D array-like (e.g., numpy.ndarray) with shape (n, dim).

    Returns:
        Any: The input normalized along axis 1, or the input unchanged if numpy
             is unavailable.
    """
    if np is None:
        return emb
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms


# ---- I/O ----------------------------------------------------------------------
def load_chunks(jsonl_path: str) -> Iterable[Dict[str, Any]]:
    """Load chunk records from a JSONL file, yielding one chunk per line.

    Skips blank lines and malformed JSON silently (continue on error).

    Args:
        jsonl_path (str): Path to a JSONL file where each line is a JSON chunk record.

    Yields:
        Dict[str, Any]: A parsed chunk dict (e.g., with keys doc_id, text, token_count).

    Raises:
        FileNotFoundError: If jsonl_path does not exist.

    Example:
        >>> for chunk in load_chunks("chunks/chunks.jsonl"):
        ...     print(chunk["doc_id"], chunk["token_count"])
    """
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


def ensure_client(persist_dir: str = "chroma_db") -> Any:
    """Return a persistent ChromaDB client, creating the persist directory if needed.

    Fails loudly if chromadb is not installed. Never falls back to in-memory storage,
    which would silently lose data between runs.

    Args:
        persist_dir (str, optional): Directory for persistent storage. Defaults to
                                     "chroma_db".

    Returns:
        Any: A chromadb.PersistentClient instance.

    Raises:
        RuntimeError: If chromadb is not installed.
    """
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Install with `pip install chromadb`")
    return chromadb.PersistentClient(path=persist_dir)


# ---- Indexing ----------------------------------------------------------------
def embed_and_index(chunks_jsonl: str = "chunks/chunks.jsonl",
                    persist_dir: str = "chroma_db",
                    collection_name: str = "documents",
                    model_name: str = "all-MiniLM-L6-v2",
                    batch_size: int = 128,
                    overwrite: bool = False,) -> Any:
    """Embed chunks from a JSONL file and index them into ChromaDB with metadata.

    Loads chunks one or more batches, encodes them via the embedding model,
    normalizes embeddings to unit length (cosine), and stores them in a persistent
    ChromaDB collection. Includes a model consistency guard: the collection records
    which model was used to build it, and refuses to index with a different model.

    Args:
        chunks_jsonl (str, optional): Path to input chunks JSONL. Defaults to
                                      "chunks/chunks.jsonl".
        persist_dir (str, optional): ChromaDB persistence directory. Defaults to
                                     "chroma_db".
        collection_name (str, optional): Name of the collection to create/use.
                                         Defaults to "documents".
        model_name (str, optional): Sentence-Transformers model identifier. Defaults
                                    to "all-MiniLM-L6-v2".
        batch_size (int, optional): Batch size for encoding. Defaults to 128.
        overwrite (bool, optional): Delete and recreate the collection if it exists.
                                    Defaults to False.

    Returns:
        Any: The ChromaDB collection object.

    Raises:
        RuntimeError: If the collection was built with a different model_name.
        FileNotFoundError: If chunks_jsonl does not exist.

    Example:
        >>> coll = embed_and_index("chunks/chunks.jsonl", overwrite=True)
        >>> print(coll.count())
        247
    """
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

    # Guard: Never mix models within one collection.
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

    # Flush any remaining items in the batch to the collection, encoding and normalizing embeddings.
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


# ---- Retrieval ---------------------------------------------------------------
def retrieve(query: str,
            persist_dir: str = "chroma_db",
            collection_name: str = "documents",
            model_name: str = "all-MiniLM-L6-v2",
            k: int = 3,
            rerank: bool = True,) -> Dict[str, Any]:
    """Retrieve top-k relevant chunks for a query using semantic search + optional reranking.

    Encodes the query, retrieves the top 20 candidates from the vector index
    (for recall), optionally reranks them via cross-encoder (for precision), and
    returns the top k. Result dicts include cosine_similarity and (if reranked)
    rerank_score fields.

    Args:
        query (str): The search query string.
        persist_dir (str, optional): ChromaDB persistence directory. Defaults to
                                     "chroma_db".
        collection_name (str, optional): Collection name to query. Defaults to
                                         "documents".
        model_name (str, optional): Embedding model (must match the index model).
                                    Defaults to "all-MiniLM-L6-v2".
        k (int, optional): Number of results to return. Defaults to 3.
        rerank (bool, optional): Apply cross-encoder reranking. Defaults to True.

    Returns:
        Dict[str, Any]: A dict with keys:
            - query (str): The input query.
            - results (List[Dict]): Top-k result dicts, each with id, document,
                                    metadata, distance, cosine_similarity, and
                                    (if reranked) rerank_score.
            - reranked (bool): Whether reranking was applied.

    Raises:
        RuntimeError: If the index was built with a different model_name.
        FileNotFoundError: If persist_dir does not exist.

    Example:
        >>> res = retrieve("off-campus housing near Penn State", k=5)
        >>> for r in res["results"]:
        ...     print(r["id"], round(r["cosine_similarity"], 2))
    """
    model = _get_model(model_name)
    client = ensure_client(persist_dir)
    collection = client.get_collection(name=collection_name)

    # Guard: Query model must match the model the index was built with.
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