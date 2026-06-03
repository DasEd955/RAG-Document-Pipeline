#!/usr/bin/env python3
"""CLI: quick retrieval against the ChromaDB index created by `embed_documents.py`."""
import argparse
import json
from pipeline.embeddings import retrieve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", help="User query string")
    p.add_argument("--persist_dir", default="chroma_db", help="ChromaDB persist directory")
    p.add_argument("--collection", default="documents", help="Chroma collection name")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model")
    p.add_argument("--k", type=int, default=3, help="Number of results to return")
    args = p.parse_args()
    res = retrieve(args.query, args.persist_dir, args.collection, args.model, args.k)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
