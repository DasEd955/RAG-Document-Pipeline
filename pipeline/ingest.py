"""Ingest.py - Document ingestion and chunking orchestration.

Loads raw HTML/text documents from a directory, cleans them via BeautifulSoup,
chunks them with the Chunker class, applies hard-floor deduplication, and
writes the result to a JSONL file. Supports adaptive chunk sizing by document
length and filters out contentless fragments.
"""
import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Iterable
from pipeline.chunker import Chunker


def load_documents_from_dir(docs_dir: Path) -> Iterable[dict]:
    """Load documents from a directory, yielding one dict per supported file.

    Yields documents in sorted order by filename. Supported formats are .txt,
    .md, .html, and .htm. Encoding errors are ignored (replaced with U+FFFD).

    Args:
        docs_dir (Path): Directory containing source documents.

    Yields:
        dict: A dict with keys:
            - doc_id (str): Stem of the filename (without extension).
            - source (str): Full path to the file as a string.
            - raw (str): Raw file contents, read with errors='ignore'.

    Raises:
        None: Non-existent or empty directories yield nothing.
    """
    """Yield {doc_id, source, raw} for each supported file in docs_dir."""
    for p in sorted(docs_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".txt", ".md", ".html", ".htm"):
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        yield {"doc_id": p.stem, "source": str(p), "raw": raw}


def ingest_and_chunk(docs_dir: str, out_path: str, chunk_size: int = 512,
                     overlap: int = 128, min_tokens: int = 100, encoding_name: str = None,
                     min_chunk_tokens: int = 50, adaptive: bool = True) -> int:
    """Clean and chunk every document in a directory, writing chunks to a JSONL file.

    Loads documents from docs_dir, cleans HTML via BeautifulSoup, chunks via the
    Chunker class, and deduplicates by content hash. Chunks below min_chunk_tokens
    are dropped (hard floor, not a soft target). Adaptive sizing per document length
    preserves narrative context in long documents (larger chunks) while pinpointing
    claims in short reviews (smaller chunks).

    Args:
        docs_dir (str): Directory containing source documents (.txt, .md, .html, .htm).
        out_path (str): Output JSONL file path (parent dir created if needed).
        chunk_size (int, optional): Target chunk size in tokens. Defaults to 512.
        overlap (int, optional): Token overlap between adjacent chunks. Defaults to 128.
        min_tokens (int, optional): Soft target minimum; chunker appends sentences.
                                    Defaults to 100.
        encoding_name (str, optional): Tiktoken encoding name (e.g., "cl100k_base").
                                       Defaults to None (auto-detect).
        min_chunk_tokens (int, optional): Hard floor: chunks below this are dropped.
                                          Defaults to 50.
        adaptive (bool, optional): Vary chunk size by document length. If False, all
                                   documents use the same chunk_size. Defaults to True.

    Returns:
        int: Total number of chunks written to out_path.

    Raises:
        None: Returns 0 if no documents found or all chunks are filtered out.

    Example:
        >>> n = ingest_and_chunk("documents", "chunks/chunks.jsonl", chunk_size=256)
        Dropped 12 chunk(s) below the 50-token floor.
        >>> print(n)
        247
    """
    docs_dir = Path(docs_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunker = Chunker(chunk_size=chunk_size, overlap=overlap, min_tokens=min_tokens,
                      encoding_name=encoding_name)

    total_chunks = 0
    dropped_short = 0
    with out_path.open("w", encoding="utf-8") as outf:
        seen_hashes = set()
        for doc in load_documents_from_dir(docs_dir):
            cleaned = chunker.clean_html(doc["raw"]) if doc["raw"].strip() else ""
            if not cleaned:
                continue

            doc_len = len(cleaned)
            if adaptive:
                # Explicit sizes so short documents actually get smaller chunks.
                """
                if doc_len > 4000:
                    doc_chunk_size, doc_overlap = max(chunk_size, 384), max(overlap, 96)
                elif doc_len > 1500:
                    doc_chunk_size, doc_overlap = 384, 96
                else:
                    doc_chunk_size, doc_overlap = 256, 64
                """
                if doc_len > 4000:
                    doc_chunk_size, doc_overlap = 256, 64
                elif doc_len > 1500:
                    doc_chunk_size, doc_overlap = 192, 48
                else:
                    doc_chunk_size, doc_overlap = 128, 32
            else:
                doc_chunk_size, doc_overlap = chunk_size, overlap

            chunks = chunker.chunk_text(cleaned, doc_id=doc["doc_id"], source=doc["source"],
                                        chunk_size=doc_chunk_size, overlap=doc_overlap,
                                        min_tokens=min_tokens) or []
            kept = 0
            for ch in chunks:
                txt = " ".join(ch.get("text", "").split())
                tok = ch.get("token_count", 0)
                # Hard Floor: drop contentless fragments. Tuned so genuine short
                # reviews survive but boilerplate stubs (e.g. "post was deleted")
                # are removed.
                if len(txt) < 40 or tok < min_chunk_tokens:
                    dropped_short += 1
                    continue
                h = hashlib.sha1(txt.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                outf.write(json.dumps(ch, ensure_ascii=False) + "\n")
                kept += 1
            total_chunks += kept

    if dropped_short:
        print(f"Dropped {dropped_short} chunk(s) below the {min_chunk_tokens}-token floor.")
    return int(total_chunks)


def _cli() -> None:
    """Command-line interface for document ingestion and chunking.

    Parses arguments for document directory, output JSONL path, chunking parameters,
    and encoding, then calls ingest_and_chunk and prints the result count.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--docs_dir", default="documents", help="Directory with source documents")
    p.add_argument("--out", default="chunks/chunks.jsonl", help="Output JSONL path")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--min_tokens", type=int, default=100, help="Soft target chunk size (chunker)")
    p.add_argument("--min_chunk_tokens", type=int, default=50, help="Hard floor: drop chunks below this")
    p.add_argument("--no_adaptive", action="store_true", help="Disable per-document adaptive sizing")
    p.add_argument("--encoding", default=None)
    args = p.parse_args()
    n = ingest_and_chunk(args.docs_dir, args.out, args.chunk_size, args.overlap,
                         args.min_tokens, args.encoding,
                         min_chunk_tokens=args.min_chunk_tokens, adaptive=not args.no_adaptive)
    print(f"Wrote {n} chunks to {args.out}")


if __name__ == "__main__":
    _cli()
