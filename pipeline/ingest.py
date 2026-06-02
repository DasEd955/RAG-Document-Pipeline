import os
import json
import argparse
from pathlib import Path
from typing import Iterable

from pipeline.chunker import Chunker


def load_documents_from_dir(docs_dir: Path) -> Iterable[dict]:
    for p in sorted(docs_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".txt", ".md", ".html", ".htm"):
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        yield {"doc_id": p.stem, "source": str(p), "raw": raw}


def ingest_and_chunk(docs_dir: str, out_path: str, chunk_size: int = 256,
                     overlap: int = 64, min_tokens: int = 50, encoding_name: str = None):
    docs_dir = Path(docs_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunker = Chunker(chunk_size=chunk_size, overlap=overlap, min_tokens=min_tokens,
                      encoding_name=encoding_name)

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as outf:
        for doc in load_documents_from_dir(docs_dir):
            cleaned = chunker.clean_html(doc["raw"]) if doc["raw"].strip() else ""
            chunks = chunker.chunk_text(cleaned, doc_id=doc["doc_id"], source=doc["source"]) or []
            for ch in chunks:
                outf.write(json.dumps(ch, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)

    return int(total_chunks)


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--docs_dir", default="documents", help="Directory with source documents")
    p.add_argument("--out", default="chunks.jsonl", help="Output JSONL path")
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--min_tokens", type=int, default=50)
    p.add_argument("--encoding", default=None)
    args = p.parse_args()
    n = ingest_and_chunk(args.docs_dir, args.out, args.chunk_size, args.overlap, args.min_tokens, args.encoding)
    print(f"Wrote {n} chunks to {args.out}")


if __name__ == "__main__":
    _cli()
