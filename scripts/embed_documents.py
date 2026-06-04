#!/usr/bin/env python3
"""CLI: embed chunks.jsonl into ChromaDB using the project's embedding pipeline."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embeddings import embed_and_index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="chunks/chunks.jsonl", help="Input chunks JSONL")
    p.add_argument("--persist_dir", default="chroma_db", help="ChromaDB persist directory")
    p.add_argument("--collection", default="documents", help="Chroma collection name")
    p.add_argument("--model", default="all-mpnet-base-v2", help="Sentence-Transformers model")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--overwrite", action="store_true", help="Delete and recreate collection if it exists")
    args = p.parse_args()
    print(f"Embedding {args.chunks} -> {args.persist_dir} (collection={args.collection})")
    embed_and_index(args.chunks, args.persist_dir, args.collection, args.model, args.batch_size, args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
