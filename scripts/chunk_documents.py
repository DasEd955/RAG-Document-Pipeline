import argparse
from pipeline.ingest import ingest_and_chunk


def main():
    p = argparse.ArgumentParser()
    # positional (kept for compatibility)
    p.add_argument("docs_dir_pos", nargs="?", default="documents", help="Positional docs dir (kept for compatibility)")
    p.add_argument("out_pos", nargs="?", default="chunks.jsonl", help="Positional out path (kept for compatibility)")
    # optional flags (preferred)
    p.add_argument("--docs_dir", dest="docs_dir_opt", default=None, help="Directory with source documents")
    p.add_argument("--out", dest="out_opt", default=None, help="Output JSONL path")
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--min_tokens", type=int, default=50)
    p.add_argument("--encoding", default=None)
    args = p.parse_args()

    docs_dir = args.docs_dir_opt or args.docs_dir_pos
    out_path = args.out_opt or args.out_pos

    n = ingest_and_chunk(docs_dir, out_path, args.chunk_size, args.overlap, args.min_tokens, args.encoding)
    print(f"Wrote {n} chunks to {out_path}")


if __name__ == "__main__":
    main()
