#!/usr/bin/env python3
"""query_documents.py - CLI for querying retrieved chunks from ChromaDB.

Performs semantic search and optional cross-encoder reranking against the indexed
chunks, printing results in JSON or human-readable debug format.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embeddings import retrieve


def main() -> None:
    """Parse query and retrieve arguments, then print results.

    Accepts a query string and optional flags for retrieval parameters (k, model,
    persist_dir, etc.). Supports --debug for human-readable output with full chunk
    text, metadata, and cosine similarity scores.
    """
    p = argparse.ArgumentParser()
    p.add_argument("query", help="User query string")
    p.add_argument("--persist_dir", default="chroma_db", help="ChromaDB persist directory")
    p.add_argument("--collection", default="documents", help="Chroma collection name")
    p.add_argument("--model", default="all-mpnet-base-v2", help="Sentence-Transformers model")
    p.add_argument("--k", type=int, default=3, help="Number of results to return")
    p.add_argument("--no-hybrid", action="store_true", help="Disable BM25 keyword recall (semantic only)")
    p.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    p.add_argument("--source", default=None, help="Only return chunks whose source contains this substring")
    p.add_argument("--min-rating", type=float, default=None, help="Minimum rating (requires a rating metadata field)")
    p.add_argument("--date-after", default=None, help="Only chunks with date >= this (requires a date metadata field)")
    p.add_argument("--date-before", default=None, help="Only chunks with date <= this (requires a date metadata field)")
    p.add_argument("--debug", action="store_true", help="Print full retrieved chunks, scores, and metadata for debugging")
    args = p.parse_args()

    # Assemble the metadata filter spec from any flags the user supplied.
    filters = {}
    if args.source:
        filters["source_contains"] = args.source
    if args.min_rating is not None:
        filters["min_rating"] = args.min_rating
    if args.date_after:
        filters["date_after"] = args.date_after
    if args.date_before:
        filters["date_before"] = args.date_before
    filters = filters or None

    res = retrieve(args.query, args.persist_dir, args.collection, args.model, args.k,
                   rerank=not args.no_rerank, hybrid=not args.no_hybrid, filters=filters)
    if args.debug:
        # Human-readable debug output: show full chunk text, metadata, and scores
        print(f"Query: {res.get('query')}")
        print(f"Hybrid (BM25 contributed): {res.get('hybrid')} | Reranked: {res.get('reranked')}")
        for i, r in enumerate(res.get('results', [])):
            print("\n--- Result {} ---".format(i + 1))
            dist = r.get('distance')
            meta = r.get('metadata') or {}
            doc = r.get('document') or ''
            print(f"Distance: {dist}")
            cos = r.get("cosine_similarity")
            if cos is not None:
                print(f"Cosine similarity: {cos}")
            # Hybrid retrieval scores: BM25 keyword score, fused RRF score, rerank score
            if r.get("bm25_score") is not None:
                print(f"BM25 score: {r.get('bm25_score')}")
            if r.get("rrf_score") is not None:
                print(f"RRF fused score: {r.get('rrf_score')}")
            if r.get("rerank_score") is not None:
                print(f"Rerank score: {r.get('rerank_score')}")
            # Print metadata fields of interest if present
            if meta:
                print("Metadata:")
                for k, v in meta.items():
                    print(f"  {k}: {v}")
            # Print document length and full text
            print(f"Document length (chars): {len(doc)}")
            print("\nFull chunk text:\n")
            print(doc)
            # Flag weak matches
            # Heuristics: Warn on weak matches
            try:
                if cos is not None:
                    if float(cos) < 0.7:
                        print("\nWARNING: cosine_similarity < 0.7 (weak match). Consider larger chunks or cleaner text.")
                else:
                    if dist is not None and float(dist) > 0.7:
                        print("\nWARNING: distance > 0.7 (weak match). Consider larger chunks or cleaner text.")
            except Exception:
                pass
        print()
    else:
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
