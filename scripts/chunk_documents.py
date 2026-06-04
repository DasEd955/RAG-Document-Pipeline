"""chunk_document.py - CLI wrapper for document ingestion and chunking.

Provides a command-line interface to ingest_and_chunk with both positional
(legacy) and optional flag arguments.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ingest import ingest_and_chunk


def main() -> None:
    """Parse arguments and run the chunking pipeline.

    Supports both positional arguments (for legacy compatibility) and optional
    flags (--docs_dir, --out, etc.). Optional flags take precedence over positionals.
    """
    p = argparse.ArgumentParser()
    # Positional (kept for compatibility)
    p.add_argument("docs_dir_pos", nargs="?", default="documents", help="Positional docs dir (kept for compatibility)")
    p.add_argument("out_pos", nargs="?", default="chunks/chunks.jsonl", help="Positional out path (kept for compatibility)")
    # Optional flags (preferred)
    p.add_argument("--docs_dir", dest="docs_dir_opt", default=None, help="Directory with source documents")
    p.add_argument("--out", dest="out_opt", default=None, help="Output JSONL path")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--min_tokens", type=int, default=100)
    p.add_argument("--encoding", default=None)
    args = p.parse_args()
    # Determine the document directory and output path based on optional arguments (preferred) or fallback to positional arguments
    docs_dir = args.docs_dir_opt or args.docs_dir_pos
    out_path = args.out_opt or args.out_pos

    n = ingest_and_chunk(docs_dir, out_path, args.chunk_size, args.overlap, args.min_tokens, args.encoding)
    print(f"Wrote {n} chunks to {out_path}")

# Entry point for the script
if __name__ == "__main__":
    main()
