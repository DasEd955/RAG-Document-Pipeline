"""
Embeddings.py - Hybrid retrieval over ChromaDB using Sentence-Transformers + BM25.

Public API (positional call order preserved so existing CLIs keep working):
    embed_and_index(chunks_jsonl, persist_dir, collection_name, model_name, batch_size, overwrite)
    retrieve(query, persist_dir, collection_name, model_name, k, rerank=True,
             hybrid=True, k_candidate=20, filters=None, source_boost=None)

Retrieval pipeline (Stage 4):
- Semantic recall: cosine search over mpnet embeddings returns a candidate set.
- Keyword recall (hybrid): BM25 over the same corpus catches exact entity matches
  (apartment names, addresses) that dense embeddings alone can miss.
- Fusion: the two ranked candidate lists are merged with Reciprocal Rank Fusion
  (RRF), which combines ranks rather than raw scores so cosine and BM25 score
  scales never need to be reconciled.
- Metadata filtering / boosting: candidates can be filtered (by source substring,
  date range, or minimum rating) and/or boosted (by source) before reranking.
- Rerank: a cross-encoder scores each (query, chunk) pair jointly and reorders the
  fused candidates; the top k_final are returned.

Key behaviours:
- The cross-encoder reranker actually runs now (CrossEncoder was leaking to a
  function-local before, so the rerank branch was dead code).
- The collection uses cosine space, so the returned distance IS (1 - cosine).
  No query-time re-encoding of documents.
- Models, the corpus snapshot, and the BM25 index are loaded once per process
  and cached.
- The embedding model name is recorded in collection metadata and verified at
  query time, so an index/query model mismatch fails loudly.
- No silent in-memory fallback for the Chroma client.
- Hybrid search and reranking both degrade gracefully: if rank_bm25 or the
  cross-encoder is unavailable, retrieval falls back to pure semantic search.

IMPORTANT: this version creates the collection with cosine space. If you have an
existing index built by the old (L2) code, re-run embedding with --overwrite
once, or the cosine numbers from `retrieve` will be wrong.

NOTE: date / rating filtering reads metadata fields (`date`, `rating`) that the
current ingestion pipeline does not yet extract. The filter plumbing is in place
and active, so these filters work automatically once those fields are populated;
until then they only match chunks that already carry the field.
"""
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import json
import re

# ---- Optional / Lazy Imports -------------------------------------------------
SentenceTransformer = None
CrossEncoder = None
chromadb = None
np = None
BM25Okapi = None


def _lazy_imports() -> None:
    """Import optional dependencies at runtime, setting global module handles.

    Imports SentenceTransformer, CrossEncoder, chromadb, numpy, and BM25Okapi
    (from rank_bm25). Each import is wrapped in a try-except to gracefully degrade
    if dependencies are missing. Every global must be declared before assignment
    to avoid local shadowing.
    """
    global SentenceTransformer, CrossEncoder, chromadb, np, BM25Okapi
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
    try:
        from rank_bm25 import BM25Okapi as _BM25
        BM25Okapi = _BM25
    except Exception:
        BM25Okapi = None


_lazy_imports()

# ---- Model Caching (Load Each Model Once per Process) ------------------------
_MODEL_CACHE: Dict[str, Any] = {}
_RERANKER_CACHE: Dict[str, Any] = {}
_DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Corpus snapshots and BM25 indexes, cached per collection name so the full
# document set is pulled from Chroma and tokenized only once per process.
_CORPUS_CACHE: Dict[str, Dict[str, Any]] = {}
_BM25_CACHE: Dict[str, Any] = {}

# Reciprocal Rank Fusion constant. The standard value (60) damps the influence
# of any single ranker's top positions so neither retriever dominates the fusion.
_RRF_K = 60


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


# ---- Hybrid Search: BM25 Keyword Recall + Rank Fusion ------------------------
_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25 scoring.

    A deliberately simple tokenizer (lowercase, split on non-alphanumeric runs)
    so that the same scheme is applied to both the corpus and the query. Entity
    names like "Hendricks" or "The Maxxen" survive as discrete tokens, which is
    exactly the exact-match recall BM25 is meant to provide.

    Args:
        text (str): The text to tokenize.

    Returns:
        List[str]: A list of lowercase alphanumeric tokens.

    Example:
        >>> _tokenize_for_bm25("The Maxxen, 123 Beaver Ave.")
        ['the', 'maxxen', '123', 'beaver', 'ave']
    """
    return _BM25_TOKEN_RE.findall((text or "").lower())


def _get_corpus(collection: Any) -> Dict[str, Any]:
    """Pull and cache the full set of documents + metadata from a Chroma collection.

    Fetches every document and its metadata once per process and caches the result
    keyed by collection name. This snapshot backs the BM25 index and lets fused
    candidates that surfaced only via keyword recall be resolved back to their text
    and metadata.

    Args:
        collection (Any): A Chroma collection object.

    Returns:
        Dict[str, Any]: A dict with parallel lists under keys:
            - ids (List[str]): Chunk IDs.
            - documents (List[str]): Chunk texts.
            - metadatas (List[dict]): Chunk metadata dicts.
    """
    key = collection.name
    if key not in _CORPUS_CACHE:
        got = collection.get(include=["documents", "metadatas"])
        _CORPUS_CACHE[key] = {
            "ids": got.get("ids") or [],
            "documents": got.get("documents") or [],
            "metadatas": got.get("metadatas") or [],
        }
    return _CORPUS_CACHE[key]


def _get_bm25(collection: Any) -> Optional[Any]:
    """Build or retrieve a cached BM25 index over a collection's documents.

    Tokenizes the cached corpus and constructs a BM25Okapi index, cached per
    collection name. Returns None if rank_bm25 is unavailable or the corpus is
    empty, in which case the caller falls back to pure semantic search.

    Args:
        collection (Any): A Chroma collection object.

    Returns:
        Optional[Any]: A BM25Okapi index, or None if unavailable.
    """
    if BM25Okapi is None:
        return None
    key = collection.name
    if key not in _BM25_CACHE:
        corpus = _get_corpus(collection)
        tokenized = [_tokenize_for_bm25(d or "") for d in corpus["documents"]]
        try:
            _BM25_CACHE[key] = BM25Okapi(tokenized) if tokenized else None
        except Exception:
            _BM25_CACHE[key] = None
    return _BM25_CACHE[key]


def _rrf_fuse(rankings: List[List[str]], k_rrf: int = _RRF_K) -> Dict[str, float]:
    """Fuse multiple ranked ID lists into a single score map via Reciprocal Rank Fusion.

    Each ranking contributes 1 / (k_rrf + rank) to every ID it contains (rank is
    0-based). Combining ranks rather than raw scores avoids having to reconcile the
    different score scales of cosine similarity and BM25.

    Args:
        rankings (List[List[str]]): A list of ranked ID lists (best first), one per
                                    retriever.
        k_rrf (int, optional): The RRF damping constant. Defaults to _RRF_K (60).

    Returns:
        Dict[str, float]: A map from ID to fused score (higher is better).

    Example:
        >>> _rrf_fuse([["a", "b"], ["b", "c"]])  # doctest: +SKIP
        {'a': 0.0164, 'b': 0.0325, 'c': 0.0161}
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k_rrf + rank + 1)
    return scores


# ---- Metadata Filtering & Source Boosting ------------------------------------
def _passes_metadata_filter(meta: Optional[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> bool:
    """Test whether a chunk's metadata satisfies an optional filter spec.

    Supported filter keys (all optional; absent keys are not enforced):
        - source_contains (str): case-insensitive substring match on `source`.
        - equals (dict): exact-match requirements on arbitrary metadata fields.
        - date_after / date_before (str): inclusive bounds compared against a
          `date` metadata field (lexicographic, so ISO-8601 dates sort correctly).
        - min_rating (float): minimum value for a numeric `rating` metadata field.

    A chunk that lacks a field referenced by an active date/rating filter fails the
    filter (it cannot be shown to satisfy a constraint it has no data for).

    Args:
        meta (Optional[Dict[str, Any]]): The chunk's metadata, or None.
        filters (Optional[Dict[str, Any]]): The filter spec, or None for no filtering.

    Returns:
        bool: True if the chunk passes (or no filters were given), False otherwise.
    """
    if not filters:
        return True
    meta = meta or {}

    src = filters.get("source_contains")
    if src and src.lower() not in str(meta.get("source", "")).lower():
        return False

    for field, expected in (filters.get("equals") or {}).items():
        if meta.get(field) != expected:
            return False

    date_after = filters.get("date_after")
    date_before = filters.get("date_before")
    if date_after or date_before:
        d = meta.get("date")
        if d is None:
            return False
        if date_after and str(d) < str(date_after):
            return False
        if date_before and str(d) > str(date_before):
            return False

    min_rating = filters.get("min_rating")
    if min_rating is not None:
        r = meta.get("rating")
        try:
            if r is None or float(r) < float(min_rating):
                return False
        except (TypeError, ValueError):
            return False

    return True


def _apply_source_boost(fused: Dict[str, float], records: Dict[str, Dict[str, Any]],
                        source_boost: Optional[Dict[str, float]]) -> None:
    """Multiply fused scores in place for chunks whose source matches a boost rule.

    Lets the caller up- or down-weight whole sources (e.g., trust the official
    Penn State guide over an anonymous forum post) without discarding anything.
    Matching is case-insensitive substring on the `source` metadata field; the
    first matching rule per chunk is applied.

    Args:
        fused (Dict[str, float]): ID -> fused score, modified in place.
        records (Dict[str, Dict[str, Any]]): ID -> result record (with "metadata").
        source_boost (Optional[Dict[str, float]]): Map of source substring ->
            multiplier (e.g., {"livingoffcampus.psu.edu": 1.5}). None disables boosting.

    Returns:
        None: `fused` is mutated in place.
    """
    if not source_boost:
        return
    for id_, score in list(fused.items()):
        src = str(((records.get(id_) or {}).get("metadata") or {}).get("source", "")).lower()
        for substring, multiplier in source_boost.items():
            if substring.lower() in src:
                fused[id_] = score * float(multiplier)
                break


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
def _semantic_recall(collection: Any, model: Any, query: str,
                     candidate_n: int) -> List[Dict[str, Any]]:
    """Run cosine semantic search and return candidate records in ranked order.

    Encodes and normalizes the query, queries the cosine-space collection, and maps
    each hit into a result record. Because the collection is cosine space, the
    returned distance equals 1 - cosine_similarity.

    Args:
        collection (Any): A Chroma collection object.
        model (Any): The loaded SentenceTransformer embedding model.
        query (str): The search query string.
        candidate_n (int): Number of candidates to request from the vector store.

    Returns:
        List[Dict[str, Any]]: Ranked result records, each with keys id, document,
            metadata, distance, cosine_similarity.
    """
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q_emb = _normalize(q_emb)
    q_list = q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb)

    res = collection.query(
        query_embeddings=q_list,
        n_results=candidate_n,
        include=["metadatas", "documents", "distances"],
    )
    docs_list = (res.get("documents") or [[]])[0]
    metas_list = (res.get("metadatas") or [[]])[0]
    dists_list = (res.get("distances") or [[]])[0]
    ids_list = (res.get("ids") or [[]])[0]

    records: List[Dict[str, Any]] = []
    for i in range(len(docs_list)):
        dist = dists_list[i] if i < len(dists_list) else None
        cos = (1.0 - dist) if dist is not None else None
        records.append({
            "id": ids_list[i] if i < len(ids_list) else None,
            "document": docs_list[i],
            "metadata": metas_list[i] if i < len(metas_list) else None,
            "distance": dist,
            "cosine_similarity": cos,
        })
    return records


def _bm25_recall(collection: Any, query: str, candidate_n: int) -> List[Dict[str, Any]]:
    """Run BM25 keyword search and return candidate records in ranked order.

    Scores the entire cached corpus against the tokenized query, keeps the top
    candidate_n with a positive score, and builds result records from the corpus
    snapshot. Records carry a bm25_score but no cosine_similarity (these candidates
    were found by keyword overlap, not the vector index).

    Args:
        collection (Any): A Chroma collection object.
        query (str): The search query string.
        candidate_n (int): Maximum number of keyword candidates to return.

    Returns:
        List[Dict[str, Any]]: Ranked result records, each with keys id, document,
            metadata, bm25_score (cosine_similarity/distance are None). Empty if
            BM25 is unavailable.
    """
    bm25 = _get_bm25(collection)
    if bm25 is None:
        return []
    corpus = _get_corpus(collection)
    ids = corpus["ids"]
    if not ids:
        return []

    scores = bm25.get_scores(_tokenize_for_bm25(query))
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)

    records: List[Dict[str, Any]] = []
    for i in order[:candidate_n]:
        if scores[i] <= 0:
            break  # ranked descending: once non-positive, the rest are too
        records.append({
            "id": ids[i],
            "document": corpus["documents"][i] if i < len(corpus["documents"]) else "",
            "metadata": corpus["metadatas"][i] if i < len(corpus["metadatas"]) else None,
            "distance": None,
            "cosine_similarity": None,
            "bm25_score": float(scores[i]),
        })
    return records


def retrieve(query: str,
            persist_dir: str = "chroma_db",
            collection_name: str = "documents",
            model_name: str = "all-MiniLM-L6-v2",
            k: int = 3,
            rerank: bool = True,
            hybrid: bool = True,
            k_candidate: int = 20,
            filters: Optional[Dict[str, Any]] = None,
            source_boost: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Retrieve top-k chunks via hybrid semantic + keyword search, then rerank.

    Pipeline: semantic (cosine) recall and, when hybrid is enabled, BM25 keyword
    recall both produce candidate lists; these are fused with Reciprocal Rank
    Fusion. Candidates are then optionally filtered (source/date/rating) and
    boosted (source), and finally reordered by a cross-encoder reranker. The top k
    are returned. Each stage degrades gracefully: missing BM25 or cross-encoder
    dependencies fall back to the available signal.

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
        hybrid (bool, optional): Fuse BM25 keyword recall with semantic recall.
                                 Defaults to True.
        k_candidate (int, optional): Candidate pool size per retriever before
                                     fusion/rerank. Defaults to 20.
        filters (dict, optional): Metadata filter spec passed to
            _passes_metadata_filter (keys: source_contains, equals, date_after,
            date_before, min_rating). Defaults to None (no filtering).
        source_boost (dict, optional): Map of source substring -> score multiplier
            applied to fused scores. Defaults to None (no boosting).

    Returns:
        Dict[str, Any]: A dict with keys:
            - query (str): The input query.
            - results (List[Dict]): Top-k result records, each with id, document,
                metadata, distance, cosine_similarity, rrf_score, optional
                bm25_score, and (if reranked) rerank_score.
            - reranked (bool): Whether cross-encoder reranking was applied.
            - hybrid (bool): Whether BM25 keyword recall actually contributed.

    Raises:
        RuntimeError: If the index was built with a different model_name.
        FileNotFoundError: If persist_dir does not exist.

    Example:
        >>> res = retrieve("Hendricks Investments reviews", k=5,
        ...                source_boost={"reddit.com": 1.2})
        >>> for r in res["results"]:
        ...     print(r["id"], round(r["rrf_score"], 4))
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

    # When a filter is active we may discard many candidates, so widen the recall
    # pool to keep enough survivors for reranking.
    pool_n = max(k, k_candidate)
    recall_n = pool_n * 3 if filters else pool_n

    # ---- Recall: semantic (always) + BM25 keyword (hybrid) -------------------
    semantic = _semantic_recall(collection, model, query, recall_n)
    keyword = _bm25_recall(collection, query, recall_n) if hybrid else []
    used_hybrid = bool(keyword)

    # Build one record per unique id; prefer the semantic record (it carries the
    # cosine score) but fold in the bm25_score from the keyword record.
    records: Dict[str, Dict[str, Any]] = {}
    for r in semantic:
        if r["id"] is not None:
            records[r["id"]] = r
    for r in keyword:
        if r["id"] is None:
            continue
        if r["id"] in records:
            records[r["id"]]["bm25_score"] = r.get("bm25_score")
        else:
            records[r["id"]] = r

    if not records:
        return {"query": query, "results": [], "reranked": False, "hybrid": used_hybrid}

    # ---- Fusion: combine the two rankings with RRF ---------------------------
    rankings = [[r["id"] for r in semantic if r["id"] is not None]]
    if keyword:
        rankings.append([r["id"] for r in keyword if r["id"] is not None])
    fused = _rrf_fuse(rankings)

    # ---- Metadata filtering + source boosting --------------------------------
    candidate_ids = [cid for cid in fused if _passes_metadata_filter(records[cid].get("metadata"), filters)]
    _apply_source_boost(fused, records, source_boost)

    # Rank candidates by fused (and possibly boosted) score, trim to the pool.
    candidate_ids.sort(key=lambda cid: fused.get(cid, 0.0), reverse=True)
    candidate_ids = candidate_ids[:pool_n]

    results: List[Dict[str, Any]] = []
    for cid in candidate_ids:
        rec = records[cid]
        rec["rrf_score"] = fused.get(cid, 0.0)
        results.append(rec)

    if not results:
        return {"query": query, "results": [], "reranked": False, "hybrid": used_hybrid}

    # ---- Rerank: cross-encoder precision pass --------------------------------
    reranker = _get_reranker() if rerank else None
    if reranker is not None:
        pairs = [[query, r["document"] or ""] for r in results]
        scores = reranker.predict(pairs)
        scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda r: r.get("rerank_score", float("-inf")), reverse=True)
    else:
        # No reranker: fused RRF order is already the best available ranking.
        results.sort(key=lambda r: r.get("rrf_score", 0.0), reverse=True)

    return {
        "query": query,
        "results": results[:k],
        "reranked": reranker is not None,
        "hybrid": used_hybrid,
    }