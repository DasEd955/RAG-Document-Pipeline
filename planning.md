# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

&emsp;The domain for elected pursuement for the RAG Document Pipeline will be off-campus housing reviews & relevant safety information, such as reports of student apartments through forums like Reddit, CollegeConfidential, and/or Yelp reviews. 
     
&emsp;This knowledge is crucial & often hard to find due to the practical considerations of students seeking housing for the first time independently, along with the often vested interest of universities. In many institutions, such as the author's own Pennsylvania State University, housing demand far outnumbers available supply; the race to secure off-campus housing starts as early as a year prior. In addition, for many students, seeking off-campus housing may often be their first experience with factors such as leasing agreements, private property owners, and fiscal responsibility to a non-educational institution. 
     
&emsp;While it is the author's belief that most landlords are fair individuals who seek a return on their investment while providing a service to their communities, it is inevitable that in many large college towns, there exists some unscrupulous actors who fail to maintain their property, embed predatory/unenforceable clauses into the leasing agreement, or infringe on privacy such as entering without notifying a tenant. 
     
&emsp;In these situations, ignorance is often preyed upon by bad actors. Students who have only been exposed to educational institutions may falsely presume that private landlords will have their best interest at heart when signing a contract. Furthermore, in a world where even many older adults fail to read a contract carefully or have education in basic contract law, it should be no surprise that many students may fail to do so, signing predatory clauses. In addition, students may come from backgrounds where they are first-generation college students, first-time renters, or have no familial backing to help them navigate real estate contracts. 
     
&emsp;In addition, universities often have a vested financial interest, bar massive infringements, to avoid calling out these bad actors on official channels of communication. In many college towns, the institution is often one of the largest property portfolio holders. Publicly identifying bad landlords can result in devaluing surrounding real estate, spark controversy with influential donors, or hinder municipality relations. 
     
&emsp;The scope of the deficit is why access to this knowledge via unofficial forums of communication is crucial to student equity & fairness. As highlighted above, students who are from disadvantaged/non-historically represented backgrounds are often the most vulnerable to predatory practices in off-campus housing. It is the author's opinion that equalizing access to information is a key factor in driving equality. The purpose of this project shall be a case demo that tests the hypothesis that with access to information, students of all backgrounds will be able to make an educated choice in their best interests that allows them to secure safe, accessible housing from fair-practice property owners.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source              |     Description       |   URL or location    |
|---|---------------------|-----------------------|----------------------|
| 1 | Reddit              |    Forum Post         | https://www.reddit.com/r/PennStateUniversity/comments/1rhinfv/apartment_recommendations/ |
| 2 | Reddit              |    Forum Post         | https://www.reddit.com/r/PennStateUniversity/comments/13hyekl/looking_for_housing_around_state_college/|
| 3 | Tripadvisor         |   Aggregation Page    | https://www.tripadvisor.com/ShowTopic-g53755-i1075-k6936316-Off_Campus_Housing-State_College_Pennsylvania.html|
| 4 | Penn State          | Official Campus Guide to Off-Campus Living | https://livingoffcampus.psu.edu/resources/article/1830-when-the-lease-starts |
| 5 | College Magazine    |    Top 10 Rating      | https://www.collegemagazine.com/ten-best-places-to-live-off-campus-at-penn-state/ |
| 6 | The Daily Collegian |    News Article       | https://www.psucollegian.com/news/penn-state-1st-year-students-discuss-finding-off-campus-housing-apartments/article_783cc63c-e4d6-11ef-8e8c-3fe7e0db1f05.html|
| 7 | The Daily Collegian |    News Article       | https://www.psucollegian.com/news/we-wanted-to-be-downtown-students-share-the-factors-behind-their-off-campus-housing-choices/article_b21d3473-49a7-4afe-b42f-bf7038d8baff.html |
| 8 | Facebook            |      Group Post       | https://www.facebook.com/groups/669735056706459/posts/2600632426950036/ |
| 9 | Reddit              |    Forum Post         | https://www.reddit.com/r/PennStateUniversity/comments/1i2u740/off_campus_housing/ |
| 10 | Reddit             |    Forum Post         | https://www.reddit.com/r/PennStateUniversity/comments/1jzuq7t/reviews_on_these_offcampus_apartments/ |
| 11 | Reddit             |    Forum Post         | https://www.reddit.com/r/PennStateUniversity/comments/hglm4y/worst_landlord_in_town/ |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

&emsp;Chunk size & overlap are chosen adaptively per document. We use the cleaned character length as a proxy for document type. Documents are not classified (i.e., forum post, article) explicitly; length is employed as a reliable substitute for the scope of this corpus. 

**Chunk Size & Overlap:**

| # | Document Length (Cleaned Chars) | Expected Content              | Chunk Size    | Overlap |
|---|---------------------------------|-------------------------------|---------------|---------|
| 1 | >4,000                          | News articles, long guides    | 256           | 64      |
| 2 | 1,500-4,000                     | Forum threads, short articles | 192           | 48      |
| 3 | <1,500                          | Single reviews, brief posts   | 128           | 32      |

**Size Floor (Dual Mechanism):**
 - Soft Target: When a Greedy-accumulated chunk is undersized, the chunker appends the next sentence before emitting, so chunks are not cut off prematurely.
 - Hard Floor (50 Tokens): Any chunk still below 50 tokens is dropped & not embedded. This removes contentless fragments that would otherwise score deceptively high upon keyword overlap.   

**Reasoning:**
- Strategy (Sentence-First & Token Capped): Documents are cleaned, split into paragraphs, then sentence-tokenized. Chunks are built by Greedily accumulating whole sentences until the token cap is reached, which preserves coherent assetions & avoids mid-sentence cuts. A sentence longer than the cap is split token-wise as a fallback mechanism. 

- Preprocessing: HTML is normalized with BeautifulSoup (i.e., scripts/nav/footer etc. removed). Residual inline boilerplate artifacts are then scrubbed from the body text (i.e., social media share buttons, reaction counts, relative timestamps, etc.). Phrases that are immediately repeated are collapsed, a paragraph info density filter drops low-value lines, and chunks that are exact duplicates are removed via a content hash. 

- Size Rationale: Longer documents get smaller chunks (256) to keep narrative & causal context intact. Shorter documents get smaller chunks (128) to pinpoint specific claims & reduce noise. The middle tier (192 tokens) keeps medium length documents from being forced to either extreme. 

- Overlap Purpose: A 64-token overlap ensures that facts near a chunk boundary will survive intact in at least one chunk. A 32-token boundary is employed sufficiently for shorter chunks. 

- Token Counting: Token counts are measured with tiktoken (cl100k_base), a BPE library, as a stable proxy. It should be noted that this is not the embedding model's exact tokenizer, so counts are approximate budgeting figures rather than exact model tokens. All caps (<256) stay comfortably within the embedding model's 384-token maximum sequence length. No chunk is silently truncated at embed time. 

- Fallback Mode: Only if tiktoken is unavailable, we fall back to crude whitespace word counts & word-based splitting. Tokens counts are preferred whenever available. 

- Metadata Logic: Each chunk record carries doc_id, source, chunk_index, char_span, token_span, and token_count. The fields pushed to the vector store are doc_id, source, chunk_index, char_span, and token_count, ensuring enough context for precise grounding & accurate source attribution. 

- Cost & Performance: Heuristic token-aware chunking controls embedding cost & LLM context while preserving semantics. Paragraph-first logic & exact duplicate chunk removal reduces redundancy of chunks and improves attribution. 

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding Model:**
- `all-mpnet-base-v2` via `sequence-transformers`: Runs locally with no API key or rate limits, which keeps the project on a free stack. 768-dimensional, 384-token maximum sequence.
- Chosen over the recommended `all-MiniLM-L6-v2` as it handles paraphrased, noisy, opinion-heavy user text better. In addition, its 384-token window leaves headroom over our specified 256-token chunk cap, whereas MiniLM's 256-token cap would sit right at the edge. 
- Embeddings are L2-normalized & stored in a ChromaDB collection configured for cosine space, so that the returned distance equals `1 - cosine_similarity`.
- A Model Consistency Guard records the embedding model name in the collection metadata & refuses to query an index that was built with a different model. This prevents silent dimension and/or space mismatches. 
- Token counts for chunk budgeting use `tiktoken (cl100k_base)` as a proxy; this is not mpnet's tokenizer, so counts are approximate (reference Chunking Strategy).

**Top-K Retrieval Counts:**
- Candidate Retrieval: `k_candidate = 20` (fast semantic recall to build a candidate set).
     - Note: Rerank with cross-encoder when precision is needed; this improves accuracy at the tradeoff of higher computing cost.
- Rerank & pass `k_final = 3–5` to the generator: `k_final = 3` for longer (256 token) chunks; `k_final = 5` for shorter (128 token) chunks.
     -- This ensures for longer chunks, context is preserved; shorter chunks gain precision
- Large aggregation or sentiment queries may need `k_final = 6–8`.
- Note: Stop concatenating chunks at generation time when adding another would exceed the LLM token budget.
     - Mathematically: `sum(token_chunks) + prompt_tokens > model_context`

**Retrieval strategy (Hybrid Recall + RRF Fusion + Cross-Encoder Rerank):**
- Semantic Recall: Cosine semantic search over mpnet embeddings returns the top `k_candidate = 20`.
- Keyword Recall (Hybrid): A BM25 index (`rank_bm25`) over the same chunk corpus scores the query by exact term overlap, returning its own top `k_candidate`. This catches exact entity matches (apartment names like "The Maxxen", landlord names like "Hendricks", addresses) that dense embeddings alone can miss when the surrounding semantics are weak.
- Fusion: The semantic and BM25 candidate rankings are merged with Reciprocal Rank Fusion (`RRF`, `k = 60`). RRF combines ranks rather than raw scores, so the incompatible scales of cosine similarity and BM25 never need to be reconciled, and an item ranked highly by both retrievers rises to the top.
- Rerank: A lightweight cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each `(query, chunk)` pair jointly & reorders the fused candidates; the top-k_final are returned. This is the precision step & is the mechanism that corrects cases where a short, keyword-dense chunk inaccurately outscores the true answer on raw cosine. 
- Graceful Degradation: If `rank_bm25` is unavailable, retrieval falls back to pure semantic recall; if the cross-encoder is unavailable, candidates are returned in fused RRF order. The response reports `hybrid` (did BM25 contribute) and `reranked` flags.

**Metadata Filtering & Boosting (Implemented):**
- Filtering: `retrieve(...)` accepts a `filters` spec applied to the candidate pool before reranking. Supported keys: `source_contains` (case-insensitive substring on `source`), `equals` (exact match on arbitrary fields), `date_after` / `date_before` (inclusive ISO-date bounds), and `min_rating` (numeric floor). When a filter is active, the recall pool is widened so enough candidates survive for reranking.
- Boosting: `source_boost` maps a source substring to a score multiplier (e.g., trust the official Penn State guide over an anonymous forum post), applied to the fused scores without discarding non-matching chunks.
- `source` filtering/boosting works today against the `source` metadata already stored on every chunk. `date` and `rating` filtering plumbing is active but depends on those fields being populated, which requires extending extraction (see below); until then those filters only match chunks that already carry the field.

**Conversational Memory (Implemented):**
- Goal: support multi-turn queries where a context-dependent follow-up (e.g., "Is it expensive?", "What about parking there?") is understood against the previous turns.
- Query Condensation: before retrieval, `condense_question(...)` folds the recent conversation plus the latest message into one standalone search query via the LLM, resolving pronouns/ellipsis and carrying over constraints. This is essential because a bare follow-up like "Is it expensive?" embeds poorly: the entity it refers to lives in an earlier turn. Measured example: the follow-up "Is it expensive?" after asking about The Maxxen condenses to "Is The Maxxen expensive?", which then retrieves the $1,400/month chunks.
- History-Aware Generation: the recent turns are inserted into the generation prompt for reference resolution only. The grounding contract is explicitly reasserted in the prompt: prior turns are NOT a source of facts, the answer must still come solely from the numbered passages, and the refusal path is unchanged.
- State: a `Conversation` class (`app/conversation.py`) holds a rolling, bounded history (`max_history_turns`, default 6) and forwards it to `ask(...)` each turn; the Gradio UI keeps one `Conversation` per browser session via `gr.State`, and a terminal REPL is available via `python -m app.conversation`.
- Grounding Note: condensation adds one short LLM call per follow-up (the first turn has no history and is unchanged). The "no answer without context" guarantee is preserved: empty retrieval still refuses before any answer is generated.

**Planned Stretch (Not Implemented Yet)**:
- Extended Extraction: Parse and store `date` (post/article date) and `rating` (e.g., star/score) at ingestion so the existing `date`/`rating` filters become fully effective across the corpus.

**Production Tradeoffs & Reflection:**

- If cost were no object, the preferred choice would be an embedding model with long context, multilingual support, domain fine-tuning for real estate specific terms, and low-latency hosted or on-prem inference (e.g., a hosted model such as `text-embedding-3-large`, or a model fine-tuned on housing & legal texts).
- Tradeoff Considerations: Accuracy vs. cost vs. latency vs. privacy (hosted API vs self-host). A local model such as mpnet wins on privacy & cost; a hosted model may win on raw accuracy & maintenance burden.
- Architecture Pattern: Cheap semantic index for recall + cross-encoder reranker for precision; cache embeddings and precompute reranks for popular queries.

**Evaluation & Tuning (Planned):**

- Experiment Grid: Chunk sizes (128, 256, 384) × `k_final` (3, 5, 8).
- Metrics: `recall@k`, `MRR`, final-answer accuracy, hallucination rate, and latency/cost per query.
- Current Verification is Manual: `scrips/query_documents.py ... --debug` prints the retrieved chunk text, distance, cosine similarity, and rerank score per query; `recall@k`/`MRR` are planned for the formal evaluation report. 

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about Hendricks Investments properties? | Negative sentiment: students report management problems & advise avoiding Hendricks Investments. |
| 2 | Is downtown State College, PA expensive? | Yes, students report rental costs are higher than alternatives. |
| 3 | Do you have to act fast to get off-campus housing at Penn State? | Yes, first year students often lease for their second year during fall semester. |
| 4 | Is The Maxxen well ranked by students? | Positive Sentiment: Students describe The Maxxen as luxury furnished & generally safe. |
| 5 | Can I feasibly live off-campus with no car? | Yes, students report that State College has reliable public transport. |

**Scoring Rubric:**
- 3 (Highest): Claim supported & at least one cited chunk or clear reference from the corpus.
- 2: Correct claim but missing citation or nuance. 
- 1: Contradicts corpus, minor hallucinations, or utilizes unsupported facts. 
- 0 (Lowest): Complete chunk generation failure

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Missing/Broken Attribution: Web scrapers or HTML normalization can result in lost URLs, headings, or paragraph offsets, leading to chunking inconsistencies.
     - Mitigation Approach: Preserve the source URL, store character/token span, include short excerpt and/or heading with every chunk so answers can cite exact provenance. 

2. Noisy/Off-Topic Content: Long news articles & forum pages may include HTML elements for navigation, ads, signatures, or filler content that inflate index size & produce irrelevant retrievals. 
     - Mitigation Approach: Run boilerplate removal, dedupe near duplicates, apply an info-density filter (i.e., a small classification model) before embedding, and/or skip low value paragraphs. 

3. Key Context/Facts Split Across Chunks: Fixed splits can divide a factual claim or context from its justification.
     - Mitigation: Implement sentence-aware chunking + overlap, enforce a minimum chunk size & merge tiny fragments, unit test with targeted queries to ensure facts appear intact in at least one chunk. 

4. Privacy & Legal Risk: Forum posts may contain copyrighted content. 
     - Mitigation: Redact obvious PII, document data usage policy, prefer linking to sources rather than publication of full verbatim excerpts. 
---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

![RAG Architecture Diagram](RAG-Architecture.png)

**Pipeline Stages:**
1. Document Ingestion (`requests`/Playwright → local HTML) → 
2. Chunking (`chunker.py`: `BeautifulSoup` cleaning + sentence-first, `tiktoken` counted chunks) → 
3. Embedding + Vector Store (`all-mpnet-base-v2` via `sentence-transformers` →  `ChromaDB`, cosine space) ->
4. Retrieval (hybrid: cosine + `rank_bm25` keyword recall, top-20 each → RRF fusion → metadata filter/boost → `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank → top-k) → 
5. Generation (Groq `llama-3.3-70b-versatile`, grounded prompt, multi-turn conversational memory via query condensation, `gradio` chat interface)

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion & Chunking:**
- AI Tools: GitHub Copilot (in-editor coding), Claude (design prompting & QA).
- Input to AI: `planning.md` Chunking Strategy, Documents List, RAG Architecture png, 2-3 Representative Raw Documents/Sample Corpus.
- Expected Output: `scripts/fetch_documents.py` (load URLs → local HTML), `pipeline/chunker.py` (BeautifulSoup `clean_html` plus the sentence-first `chunk_text(...)` returning chunks + metadata), and `pipeline/ingest.py` + `scripts/chunk_documents.py` (orchestration, adaptive sizing, dedup, JSONL output). Cleaning lives inside `chunker.py` rather than a separate clean.py.
- Verification: Print 5 random chunks & run tests to check no empty chunks, sentence boundary alignment, and overlap correctness. Assert that total chunk count is in expected range (50-2000).

**Milestone 4 — Embedding & Retrieval:**
- AI Tools: GitHub Copilot (in-editor coding), Claude (design prompting & QA), `sentence-transformers` (API embeddings), `ChromaDB`/`FAISS` (vector store).
- Input to AI: `planning.md` Retrieval Approach, sample chunk JSON, and RAG Architecture png.
- Expected Output: A single `pipeline/embeddings.py` `containing embed_and_index(...)` (embed + push to ChromaDB with metadata, cosine space, model-consistency guard) and `retrieve(...)` (cosine recall + cross-encoder rerank), `driven by scripts/embed_documents.py` and `scripts/query_documents.py`. Embedding, indexing, retrieval, and reranking are consolidated in one module rather than separate embed.py / index.py / retrieve.py / rerank.py files.
- Verification: Run `scripts/query_documents.py` on 3 evaluation queries with `--debug`; inspect top-k chunks, distances, cosine similarity, and rerank scores; expect top cosine distances below ~0.5 for good matches. (`recall@k`/`MRR` planned for the evaluation report.)

**Milestone 5 — Generation and interface:**
- AI Tools: GitHub Copilot (in-editor coding), Claude (design prompting & QA), Groq Wrapper
- Input to AI: `planning.md` Grounding Rules, `retrieve(...)` outputs, RAG Architecture png. 
- Expected Output: a `query.py` that composes a grounding prompt (e.g., context + strict instruction), calls the LLM, and returns answer + sources; app.py minimal Gradio UI & integration tests tests/test_end2end.py
     - Example Grounding Prompt: "Use only provided context; if insufficient, say 'I don't have enough info'"
- Verification: End-to-End tests for 2-3 queries; responses must cite source document names for grounding test & when out-of-scope, return the refusal phrase. Manually check at minimum one correct, one partial, one failure case & record in README.md. 

**Note to Self (Delete Later): Commit Checkpoints**
- Commit after each milestone with these files:
     - Milestone 3: ingest.py, clean.py, chunker.py, tests/test_chunker.py, requirements.txt.
     - Milestone 4: embed.py, index.py, retrieve.py, rerank.py, scripts/test_retrieval.py.
     - Milestone 5: query.py, app.py, tests/test_end2end.py, README updates.
- Add short CI/local test script to run unit + basic integration tests.