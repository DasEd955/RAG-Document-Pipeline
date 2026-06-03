import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Iterable
from pipeline.chunker import Chunker


def load_documents_from_dir(docs_dir: Path) -> Iterable[dict]:
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
                     min_chunk_tokens: int = 50, adaptive: bool = True):
    """Clean + chunk every document and write chunks to a JSONL file.

    min_chunk_tokens is a HARD floor: any chunk below it is dropped. This is the
    fix for the "deleted Reddit post" problem -- the chunker's min_tokens is only
    a soft target (it can't grow a one-sentence document), so contentless stubs
    used to survive and outrank real answers. The floor removes them here.

    adaptive=True genuinely varies chunk size by document length (long guides get
    big chunks for context; short forum posts get smaller chunks for specificity).
    The old `max(chunk_size, ...)` form could only ever increase size, so with the
    default chunk_size every document got the same size -- it never adapted.
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
                # Hard floor: drop contentless fragments. Tuned so genuine short
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


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--docs_dir", default="documents", help="Directory with source documents")
    p.add_argument("--out", default="chunks.jsonl", help="Output JSONL path")
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
