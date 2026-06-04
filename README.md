# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

&emsp;The domain for elected pursuement for the RAG Document Pipeline will be off-campus housing reviews & relevant safety information, such as reports of student apartments through forums like Reddit, CollegeConfidential, and/or Yelp reviews. 
     
&emsp;This knowledge is crucial & often hard to find due to the practical considerations of students seeking housing for the first time independently, along with the often vested interest of universities. In many institutions, such as the author's own Pennsylvania State University, housing demand far outnumbers available supply; the race to secure off-campus housing starts as early as a year prior. In addition, for many students, seeking off-campus housing may often be their first experience with factors such as leasing agreements, private property owners, and fiscal responsibility to a non-educational institution. 
     
&emsp;While it is the author's belief that most landlords are fair individuals who seek a return on their investment while providing a service to their communities, it is inevitable that in many large college towns, there exists some unscrupulous actors who fail to maintain their property, embed predatory/unenforceable clauses into the leasing agreement, or infringe on privacy such as entering without notifying a tenant. 
     
&emsp;In these situations, ignorance is often preyed upon by bad actors. Students who have only been exposed to educational institutions may falsely presume that private landlords will have their best interest at heart when signing a contract. Furthermore, in a world where even many older adults fail to read a contract carefully or have education in basic contract law, it should be no surprise that many students may fail to do so, signing predatory clauses. In addition, students may come from backgrounds where they are first-generation college students, first-time renters, or have no familial backing to help them navigate real estate contracts. 
     
&emsp;In addition, universities often have a vested financial interest, bar massive infringements, to avoid calling out these bad actors on official channels of communication. In many college towns, the institution is often one of the largest property portfolio holders. Publicly identifying bad landlords can result in devaluing surrounding real estate, spark controversy with influential donors, or hinder municipality relations. 
     
&emsp;The scope of the deficit is why access to this knowledge via unofficial forums of communication is crucial to student equity & fairness. As highlighted above, students who are from disadvantaged/non-historically represented backgrounds are often the most vulnerable to predatory practices in off-campus housing. It is the author's opinion that equalizing access to information is a key factor in driving equality. The purpose of this project shall be a case demo that tests the hypothesis that with access to information, students of all backgrounds will be able to make an educated choice in their best interests that allows them to secure safe, accessible housing from fair-practice property owners.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

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

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

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

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model Information:**
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

**Retrieval strategy (Semantic Recall + Cross-Encoder Rerank):**
- Semantic Recall: Cosine semantic search over mpnet embeddings returns the top `k_candidate = 20`.
- Rerank: A lightweight cross-encoder `cross-encoder/ms-marco/MiniLM-L-6-v2` scores each `(query, chunk)` pair jointly & reorders the top candidates; the top-k_final are returned. This is the precision step & is the mechanism that correct cases where a short, keyword-dense chunk inaccurately outscores the true answer on raw cosine. 

**Planned Stretch (Not Implemented Yet)**:
- Hybrid Search: Add BM25/keyword recall to catch exact entity matches (i.e., apartment names, addresses) that embeddings alone can miss, then fuse with the semantic candidates before reranking. 
- Metadata Filtering: Filter/boost by `source` (implemented), and after extending extraction: by `date` and `rating`.

**Production Tradeoffs & Reflection:**

- If cost were no object, the preferred choice would be an embedding model with long context, multilingual support, domain fine-tuning for real estate specific terms, and low-latency hosted or on-prem inference (e.g., a hosted model such as `text-embedding-3-large`, or a model fine-tuned on housing & legal texts).
- Tradeoff Considerations: Accuracy vs. cost vs. latency vs. privacy (hosted API vs self-host). A local model such as mpnet wins on privacy & cost; a hosted model may win on raw accuracy & maintenance burden.
- Architecture Pattern: Cheap semantic index for recall + cross-encoder reranker for precision; cache embeddings and precompute reranks for popular queries.

**Evaluation & Tuning (Planned):**

- Experiment Grid: Chunk sizes (128, 256, 384) × `k_final` (3, 5, 8).
- Metrics: `recall@k`, `MRR`, final-answer accuracy, hallucination rate, and latency/cost per query.
- Current Verification is Manual: `scrips/query_documents.py ... --debug` prints the retrieved chunk text, distance, cosine similarity, and rerank score per query; `recall@k`/`MRR` are planned for the formal evaluation report. 

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**
- The system prompt enforces hard grounding by providing the LLM with a numbered set of retrieved context passages. It instructs the model repeatedly to answer ONLY from those passages, to never use outside knowledge or invent facts, and to reply with the refusal phrase "`I don't have enough information in the loaded documents to answer that.`" when passages do not provide sufficient context. 
- System Prompt: <br><br>"You are a grounded question-answering assistant for off-campus student housing near Penn State / State College, PA. You will be given a user question and a set of numbered context passages retrieved from a fixed corpus of forum posts, reviews, news articles, and official guides. You must obey ALL of the following rules:

     1. Answer using ONLY the information contained in the numbered context passages below. The passages are your single source of truth.
     2. Do NOT use any outside or prior knowledge. Do NOT guess, extrapolate, or add any fact that is not explicitly supported by the passages.
     3. If the passages do not contain enough information to answer the question, you MUST reply with this exact sentence and nothing else:
     {refusal}
     4. When you state a fact, cite the passage number(s) it came from using square-bracket markers, e.g. [1] or [2][3]. Only ever use the [n] markers that appear in the context — never invent source names, URLs, dates, or numbers.
     5. Be concise and factual. Prefer the wording and sentiment actually expressed in the passages over your own phrasing.

     Context passages:
     {context}"
- The prompt also requires the model to cite passage numbers with square bracket notation (e.g. `[1] www.sourceurl.com`), allowing for a clean text output while providing robust grounding verification. 
"

**How source attribution is surfaced in the response:**
- Source attribution is built programatically from the retrieved chunk metadata & never parsed from model output. 
- Source attribution is cited as numbered passages with square bracket notation (e.g. `[1] www.sourceurl.com`).
- These sources are passed to the model with a separately constructed sources list mapping each [n] marker to a canonical source & similarity score. 
- If retrieval returns no results, we refuse without calling the LLM.
- If the model emits the canonical refusal phrase, we supress sources. This guarantees that citations cannot be hallucinated or omitted by the model. 

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected Answer | System Response & Sources (summarized) | Retrieval Quality | Response Accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about Hendricks Investments properties? | Negative sentiment: students report management problems & advise avoiding Hendricks Investments. | According to [1] and [2], students have expressed negative opinions about Hendricks Investments. One user stated that Hendricks Investments is "notorious in State College" and that their website and reviews on Google are "laughable" [1]. Another user mentioned that they've lived in State College for ten years and never heard anything positive about Hendricks Investments [2]. Additionally, a user described their experience with ARPM (not directly stated to be Hendricks Investments, but mentioned in the same context) as "fine if a little money hungry" and mentioned being charged for maintenance issues that weren't their fault [1][2]. <br><br> [1] 11_www.reddit.com-r-PennStateUniversity-comments-hglm4y-worst_landlord_in_town (similarity 0.53)[2] 11_www.reddit.com-r-PennStateUniversity-comments-hglm4y-worst_landlord_in_town (similarity 0.58)[3] https://www.psucollegian.com/news/penn-state-1st-year-students-discuss-finding-off-campus-housing-apartments/article_783cc63c-e4d6-11ef-8e8c-3fe7e0db1f05.html (similarity 0.39)[4] https://www.psucollegian.com/news/penn-state-1st-year-students-discuss-finding-off-campus-housing-apartments/article_783cc63c-e4d6-11ef-8e8c-3fe7e0db1f05.html (similarity 0.43)[5] https://www.psucollegian.com/news/penn-state-1st-year-students-discuss-finding-off-campus-housing-apartments/article_783cc63c-e4d6-11ef-8e8c-3fe7e0db1f05.html (similarity 0.38)| Relevant | Accurate |
| 2 | Is downtown State College, PA expensive? | Yes, students report rental costs are higher than alternatives. | According to [3] and [4], rent in downtown State College can "seem super expensive". Additionally, [5] mentions that everywhere in the area is "relatively expensive", with the user's budget being around $1,200 per month. [4] also compares the price of The Metropolitan to other newly renovated apartments, implying that prices can vary but are generally high. <br><br> [1] https://www.collegemagazine.com/ten-best-places-to-live-off-campus-at-penn-state/ (similarity 0.59)[2] https://www.collegemagazine.com/ten-best-places-to-live-off-campus-at-penn-state/ (similarity 0.60)[3] https://www.collegemagazine.com/ten-best-places-to-live-off-campus-at-penn-state/ (similarity 0.57)[4] https://www.collegemagazine.com/ten-best-places-to-live-off-campus-at-penn-state/ (similarity 0.50)[5] https://www.reddit.com/r/PennStateUniversity/comments/1rhinfv/apartment_recommendations/ (similarity 0.52) | Relevant | Accurate |
| 3 | Do you have to act fast to get off-campus housing at Penn State? | Yes, first year students often lease for their second year during fall semester. | Yes, it appears that students need to act fast to get off-campus housing at Penn State. According to [4], the search for off-campus housing in downtown State College has become a "high-stakes race" where students face pressure to commit to leases nearly a year in advance. Additionally, [4] mentions that students are "scrambling to find housing for next year" and some still don't know where they'll be living, indicating a sense of urgency. [2] also mentions that "many of the nicer, close-to-campus apartments will be all booked up" if you don't start looking early enough. [5] further supports this by stating that the process of finding off-campus housing downtown is "rushed", implying that students need to act quickly to secure a spot. <br><br> [1] https://www.facebook.com/groups/669735056706459/posts/2600632426950036/ (similarity 0.64)[2] https://www.tripadvisor.com/ShowTopic-g53755-i1075-k6936316-Off_Campus_Housing-State_College_Pennsylvania.html (similarity 0.58)[3] https://www.reddit.com/r/PennStateUniversity/comments/1jzuq7t/reviews_on_these_offcampus_apartments/ (similarity 0.58)[4] https://www.psucollegian.com/news/penn-state-1st-year-students-discuss-finding-off-campus-housing-apartments/article_783cc63c-e4d6-11ef-8e8c-3fe7e0db1f05.html (similarity 0.63)[5] https://www.psucollegian.com/news/penn-state-1st-year-students-discuss-finding-off-campus-housing-apartments/article_783cc63c-e4d6-11ef-8e8c-3fe7e0db1f05.html (similarity 0.60) | Relevant | Accurate |
| 4 | Is The Maxxen well ranked by students? | Positive Sentiment: Students describe The Maxxen as luxury furnished & generally safe. | According to the passages, The Maxxen is considered the "best apartment to live in at Penn State" [3]. Additionally, a current resident, Paige Box, is quoted as saying that the security and amenities at The Maxxen made her feel like she could "get her money's worth" [1][2]. However, another student, Ethan Conti, decided to move out of The Maxxen because he and his roommate felt they were paying too much [4][5]. Overall, the passages suggest that The Maxxen is well-regarded by students, but may be on the expensive side [1]. <br><br> [1] https://www.collegemagazine.com/ten-best-places-to-live-off-campus-at-penn-state/ (similarity 0.52)[2] https://www.collegemagazine.com/ten-best-places-to-live-off-campus-at-penn-state/ (similarity 0.51)[3] https://www.collegemagazine.com/ten-best-places-to-live-off-campus-at-penn-state/ (similarity 0.48)[4] https://www.psucollegian.com/news/we-wanted-to-be-downtown-students-share-the-factors-behind-their-off-campus-housing-choices/article_b21d3473-49a7-4afe-b42f-bf7038d8baff.html (similarity 0.37)[5] https://www.psucollegian.com/news/we-wanted-to-be-downtown-students-share-the-factors-behind-their-off-campus-housing-choices/article_b21d3473-49a7-4afe-b42f-bf7038d8baff.html (similarity 0.33)| Relevant | Accurate |
| 5 | Can I feasibly live off-campus with no car? |  Yes, students report that State College has reliable public transport. | According to [1] and [2], it may be challenging to live off-campus without a car, as it would be difficult to get to places like Walmart to get groceries. However, it is not explicitly stated that it is impossible. [2] mentions that the writer "wouldn't have been ready for an apartment as a sophomore - especially without a car", implying that having a car can be beneficial for off-campus living. On the other hand, [5] mentions students who wanted to live close to campus so they wouldn't have to rely on buses or cars, suggesting that it may be possible to live off-campus without a car if you choose a location that is close to campus and has accessible amenities. <br><br> [1] https://www.tripadvisor.com/ShowTopic-g53755-i1075-k6936316-Off_Campus_Housing-State_College_Pennsylvania.html (similarity 0.41)[2] https://www.tripadvisor.com/ShowTopic-g53755-i1075-k6936316-Off_Campus_Housing-State_College_Pennsylvania.html (similarity 0.43)[3] https://www.psucollegian.com/news/penn-state-1st-year-students-discuss-finding-off-campus-housing-apartments/article_783cc63c-e4d6-11ef-8e8c-3fe7e0db1f05.html (similarity 0.41)[4] https://www.psucollegian.com/news/penn-state-1st-year-students-discuss-finding-off-campus-housing-apartments/article_783cc63c-e4d6-11ef-8e8c-3fe7e0db1f05.html (similarity 0.42)[5] https://www.psucollegian.com/news/we-wanted-to-be-downtown-students-share-the-factors-behind-their-off-campus-housing-choices/article_b21d3473-49a7-4afe-b42f-bf7038d8baff.html (similarity 0.44) | Accurate | Relevant |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**

**What the system returned:**

**Root cause (tied to a specific pipeline stage):**

**What you would change to fix it:**

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**

**One way your implementation diverged from the spec, and why:**

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*

**Instance 2**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*
