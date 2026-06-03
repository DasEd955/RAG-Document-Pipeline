"""
Embedding and retrieval helpers using Sentence-Transformers and ChromaDB.

Implements:
- `embed_and_index(...)` : load chunks JSONL, embed with `all-MiniLM-L6-v2`, store in ChromaDB
- `retrieve(...)` : embed a query and return nearest chunks with metadata

This module keeps imports lazy so the file can be imported even if optional deps
aren't installed; functions will raise helpful errors if dependencies are missing.
"""
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional


def _lazy_imports():
    global SentenceTransformer, chromadb, Settings, np
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception:
        chromadb = None
        Settings = None
    try:
        import numpy as np
    except Exception:
        np = None


_lazy_imports()


def load_chunks(jsonl_path: str) -> Iterable[Dict[str, Any]]:
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"Chunks file not found: {jsonl_path}")
    import json
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except Exception:
                continue


def ensure_client(persist_dir: str = "chroma_db"):
    if chromadb is None or Settings is None:
        raise RuntimeError("chromadb is not installed. Install with `pip install chromadb`")
    # Use current Settings fields: request a persistent DB at `persist_dir`.
    try:
        settings = Settings(is_persistent=True, persist_directory=persist_dir)
        client = chromadb.Client(settings)
        return client
    except Exception:
        # Fallback to a default client if Settings-based creation fails for this chromadb version
        try:
            client = chromadb.Client()
            return client
        except Exception as e:
            raise RuntimeError("Failed to create chromadb client: " + str(e))


def embed_and_index(
    chunks_jsonl: str = "chunks.jsonl",
    persist_dir: str = "chroma_db",
    collection_name: str = "documents",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
    overwrite: bool = False,
) -> Any:
    """Embed chunks and index into a persistent ChromaDB collection.

    Args:
        chunks_jsonl: path to ingestion output (one JSON object per line, with `text`, `doc_id`, `chunk_index`, ...)
        persist_dir: chroma persistence directory
        collection_name: chroma collection name
        model_name: sentence-transformers model (use `all-MiniLM-L6-v2`)
        batch_size: number of texts per batch
        overwrite: if True, delete and recreate the collection

    Returns:
        The chroma collection object
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. Install with `pip install sentence-transformers`")
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Install with `pip install chromadb`")

    model = SentenceTransformer(model_name)
    client = ensure_client(persist_dir)

    # create or get collection (be conservative with API differences)
    try:
        collection = client.get_collection(name=collection_name)
        if overwrite:
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
            collection = client.create_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    ids_batch: List[str] = []
    docs_batch: List[str] = []
    met_batch: List[Dict[str, Any]] = []

    def flush():
        if not ids_batch:
            return
        emb = model.encode(docs_batch, convert_to_numpy=True, show_progress_bar=False)
        # ensure list-of-lists for chroma
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        collection.add(ids=ids_batch, documents=docs_batch, metadatas=met_batch, embeddings=emb_list)
        ids_batch.clear()
        docs_batch.clear()
        met_batch.clear()

    for item in load_chunks(chunks_jsonl):
        text = item.get("text") or ""
        doc_id = item.get("doc_id") or ""
        chunk_index = item.get("chunk_index")
        uid = f"{doc_id}__{chunk_index}"
        ids_batch.append(uid)
        docs_batch.append(text)
        met_batch.append({
            "doc_id": doc_id,
            "source": item.get("source"),
            "chunk_index": chunk_index,
            "char_span": item.get("char_span"),
            "token_count": item.get("token_count"),
        })
        if len(ids_batch) >= batch_size:
            flush()
    flush()
    try:
        client.persist()
    except Exception:
        # some chroma setups persist automatically
        pass
    return collection


def retrieve(
    query: str,
    persist_dir: str = "chroma_db",
    collection_name: str = "documents",
    model_name: str = "all-MiniLM-L6-v2",
    k: int = 3,
) -> Dict[str, Any]:
    """Return the top-k nearest chunks for `query` using the same embedding model.

    Returns the raw ChromaDB query result (ids, distances, metadatas, documents).
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. Install with `pip install sentence-transformers`")
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Install with `pip install chromadb`")

    model = SentenceTransformer(model_name)
    client = ensure_client(persist_dir)
    collection = client.get_collection(name=collection_name)
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False).tolist()
    # Use allowed include values; 'ids' is not accepted by newer chroma validators
    res = collection.query(query_embeddings=q_emb, n_results=k, include=["metadatas", "documents", "distances"])

    # Normalize into a list of result dicts for the single-query case
    out: Dict[str, Any] = {"query": query, "results": []}
    # Chroma returns lists-of-lists (one per query). We support single-query input.
    ids_list = res.get("ids", [[]])[0] if isinstance(res.get("ids"), list) else []
    docs_list = res.get("documents", [[]])[0] if isinstance(res.get("documents"), list) else []
    metas_list = res.get("metadatas", [[]])[0] if isinstance(res.get("metadatas"), list) else []
    dists_list = res.get("distances", [[]])[0] if isinstance(res.get("distances"), list) else []

    for i in range(max(len(ids_list), len(docs_list), len(metas_list), len(dists_list))):
        out["results"].append({
            "id": ids_list[i] if i < len(ids_list) else None,
            "document": docs_list[i] if i < len(docs_list) else None,
            "metadata": metas_list[i] if i < len(metas_list) else None,
            "distance": dists_list[i] if i < len(dists_list) else None,
        })

    return out
