#!/usr/bin/env python3
"""CLI: quick retrieval against the ChromaDB index created by `embed_documents.py`."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embeddings import retrieve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", help="User query string")
    p.add_argument("--persist_dir", default="chroma_db", help="ChromaDB persist directory")
    p.add_argument("--collection", default="documents", help="Chroma collection name")
    p.add_argument("--model", default="all-mpnet-base-v2", help="Sentence-Transformers model")
    p.add_argument("--k", type=int, default=3, help="Number of results to return")
    p.add_argument("--debug", action="store_true", help="Print full retrieved chunks, distances, and metadata for debugging")
    args = p.parse_args()
    res = retrieve(args.query, args.persist_dir, args.collection, args.model, args.k)
    if args.debug:
        # Human-readable debug output: show full chunk text, metadata, and distance
        print(f"Query: {res.get('query')}")
        for i, r in enumerate(res.get('results', [])):
            print("\n--- Result {} ---".format(i + 1))
            dist = r.get('distance')
            meta = r.get('metadata') or {}
            doc = r.get('document') or ''
            print(f"Distance: {dist}")
            cos = r.get("cosine_similarity")
            if cos is not None:
                print(f"Cosine similarity: {cos}")
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
            # Heuristics: warn on weak matches
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
