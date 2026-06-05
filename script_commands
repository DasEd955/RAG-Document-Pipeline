Install / update dependencies (rank-bm25 is NEW for hybrid search):

pip install -r requirements.txt
    Or just the new one: pip install "rank-bm25>=0.2.2"

NOTE: Hybrid search (BM25) needs NO re-chunk and NO re-embed. The BM25 index is
built at query time from the documents already stored in ChromaDB, so installing
rank-bm25 is enough to enable it. The full rebuild below is only needed if your
documents/ or chunks changed (e.g., after re-fetching doc 11 with Playwright).

----------------------------------------------------------------------
Full pipeline rebuild (run in order from the project root):
----------------------------------------------------------------------

Regenerate Chunks:

python -m pipeline.ingest --docs_dir documents --out chunks/chunks.jsonl

Count Chunks:

(Get-Content chunks/chunks.jsonl | Measure-Object -Line).Lines

Embed Chunks into ChromaDB:

python scripts/embed_documents.py --chunks chunks/chunks.jsonl --overwrite
    Note: the embed script is embed_documents.py (chunk_documents.py is the chunker).

----------------------------------------------------------------------
Test Queries (Stage 4 hybrid retrieval):
----------------------------------------------------------------------

Default (hybrid semantic + BM25, RRF fusion, cross-encoder rerank):

python scripts/query_documents.py "Do you have to act fast to get off-campus housing at Penn State?" --k 5 --debug

Exact-entity query — BM25 keyword recall surfaces these even at low cosine:

python scripts/query_documents.py "Hendricks Investments" --k 5 --debug

Semantic-only (ablation: disable BM25) to compare:

python scripts/query_documents.py "Hendricks Investments" --k 5 --no-hybrid --debug

Metadata filter — restrict to a single source by substring:

python scripts/query_documents.py "when does the lease start" --k 3 --source livingoffcampus.psu.edu --debug

----------------------------------------------------------------------
End-to-end grounded answer (Stage 5):
----------------------------------------------------------------------

python -m app.query "What do students say about Hendricks Investments?" --k 5
python -m app.query "when does the lease start" --k 3 --source livingoffcampus.psu.edu

----------------------------------------------------------------------
Multi-turn conversational memory (Stage 5):
----------------------------------------------------------------------

Interactive REPL — ask a question, then a follow-up that relies on it:

python -m app.conversation
    You: Is The Maxxen well ranked by students?
    You: Is it expensive?            <- condensed to "Is The Maxxen expensive?" before retrieval
    You: reset                       <- clears history;  'exit' to quit

----------------------------------------------------------------------
Run tests:
----------------------------------------------------------------------

python -m pytest tests/ -q

----------------------------------------------------------------------
Launch the app (now a multi-turn chat UI):
----------------------------------------------------------------------

python -m app.app