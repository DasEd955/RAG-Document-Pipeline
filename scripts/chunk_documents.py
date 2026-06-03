import argparse
from pipeline.ingest import ingest_and_chunk

# main(): Command-line interface function that supports both positional and optional arguments for document directory and output path, along with chunking parameters and encoding. 
    # It calls the ingest_and_chunk function and prints the total number of chunks created.
def main():
    p = argparse.ArgumentParser()
    # Positional (kept for compatibility)
    p.add_argument("docs_dir_pos", nargs="?", default="documents", help="Positional docs dir (kept for compatibility)")
    p.add_argument("out_pos", nargs="?", default="chunks.jsonl", help="Positional out path (kept for compatibility)")
    # Optional flags (preferred)
    p.add_argument("--docs_dir", dest="docs_dir_opt", default=None, help="Directory with source documents")
    p.add_argument("--out", dest="out_opt", default=None, help="Output JSONL path")
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--min_tokens", type=int, default=50)
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
