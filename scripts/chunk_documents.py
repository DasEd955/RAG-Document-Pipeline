import argparse
from pipeline.ingest import ingest_and_chunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("docs_dir", nargs="?", default="documents")
    p.add_argument("out", nargs="?", default="chunks.jsonl")
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--min_tokens", type=int, default=50)
    p.add_argument("--encoding", default=None)
    args = p.parse_args()
    n = ingest_and_chunk(args.docs_dir, args.out, args.chunk_size, args.overlap, args.min_tokens, args.encoding)
    print(f"Wrote {n} chunks to {args.out}")


if __name__ == "__main__":
    main()
