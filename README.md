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

**Sample Chunks**

Below are five representative chunks extracted from the processed corpus (each labeled with its source file). These demonstrate chunk size, coherence, and metadata alignment.

- Source: documents/01_www.reddit.com_r_PennStateUniversity_comments_1rhinfv_apartment_recommendations.html

     "Go to PennStateUniversity r/PennStateUniversity Jer_Bear_Berry Apartment Recommendations I got into the Penn State Law School at the University Park Campus. I am starting to look at house and was wondering what recommendations people have. I have heard mixed reviews and I know everywhere is relatively expensive. Places people have told me are The Yards, The Heights, and The View. These places are all within my budget ~$1,200 per month. I was wondering if people have any other recommendations and if there were any spots closer to campus that are within (or around) my price range."

- Source: documents/02_www.reddit.com_r_PennStateUniversity_comments_13hyekl_looking_for_housing_around_state_college.html

     "We have 2 cars, 2 cats, and 2 birds. (LMAO) We have a budget of around $1200. Commute radius about 30 minutes to PSU campus. 1B/2B/Studio are fine. unfurnished Her preferences: in unit W/D or hookup, big closet My preferences: garage or driveway, house or townhouse will be ideal (not necessary) Someone help me please. If anyone can find me a good place, I will buy them a big meal :/"

- Source: documents/04_livingoffcampus.psu.edu_1830-when-the-lease-starts_623e81d1.html

     "Happy Move-in Day! Here are a few move-in tips: 1. If the apartment is not clean when you arrive, contact the landlord immediately. You will not be reimbursed for cleaning the apartment yourself and will be expected to leave it in a clean condition when you move out, regardless of its condition when you moved in. 2. Before bringing your belongings into the rental unit, perform a careful walk-through inspection. Make a list of pre-existing damages."

- Source: documents/06_www.psucollegian.com_article_783cc63c-e4d6-11ef-8e8_555cba9c.html

     "The search for off-campus housing in downtown State College has become a high-stakes race, where students face skyrocketing rents, a wave of luxury apartment complexes and the pressure to commit to leases nearly a year in advance. For first-year students especially, the process can be overwhelming as they navigate not just lease terms and rental rates, but also the pressure to choose roommates and commit to living arrangements while still adjusting to college life."

- Source: documents/11_www.reddit.com-r-PennStateUniversity-comments-hglm4y-worst_landlord_in_town.html

     "I’ve been searching for 2 weeks straight to try and take my lease after the guy who was supposed to take mine found out they were doing a deal for $200 less a month. I’ve been a ball of stress this whole time, haven’t slept well, can’t sleep now at 6am. Everything about the pointe seems great. But this is so fucking scummy."

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

**Retrieval strategy (Hybrid Recall + RRF Fusion + Cross-Encoder Rerank):**
- Semantic Recall: Cosine semantic search over mpnet embeddings returns the top `k_candidate = 20`.
- Keyword Recall (Hybrid): A BM25 index (`rank_bm25`) over the same chunk corpus scores the query by exact term overlap and returns its own top `k_candidate`. This catches exact entity matches (apartment names like "The Maxxen", landlord names like "Hendricks", addresses) that dense embeddings alone can miss when the surrounding semantics are weak. Empirically, the "Hendricks Investments" chunks score only ~0.35–0.40 cosine but BM25 6.2–6.8, so keyword recall is what lifts them to the top.
- Fusion: The semantic and BM25 rankings are merged with Reciprocal Rank Fusion (`RRF`, `k = 60`), which combines ranks rather than raw scores so the incompatible cosine and BM25 scales never need reconciling.
- Rerank: A lightweight cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each `(query, chunk)` pair jointly & reorders the fused candidates; the top-k_final are returned. This is the precision step & is the mechanism that corrects cases where a short, keyword-dense chunk inaccurately outscores the true answer on raw cosine. 
- Graceful Degradation: If `rank_bm25` is unavailable, retrieval falls back to pure semantic recall; if the cross-encoder is unavailable, candidates are returned in fused RRF order. The retrieval response reports `hybrid` and `reranked` flags.

**Metadata Filtering & Boosting (Implemented):**
- Filtering: `retrieve(...)` (and `ask(...)`) accept a `filters` spec applied to the candidate pool before reranking — `source_contains` (case-insensitive substring on `source`), `equals` (exact match on arbitrary fields), `date_after` / `date_before` (inclusive ISO-date bounds), and `min_rating` (numeric floor). When a filter is active, the recall pool is widened so enough candidates survive reranking.
- Boosting: `source_boost` maps a source substring to a score multiplier (e.g., trust the official Penn State guide over an anonymous forum post), applied to fused scores without discarding non-matching chunks.
- The CLI exposes these via `--source`, `--min-rating`, `--date-after`, `--date-before`, and `--no-hybrid` / `--no-rerank`.
- `source` filtering/boosting works today against the `source` metadata stored on every chunk. `date`/`rating` filter plumbing is active but depends on those fields being populated, which requires extending extraction.

**Planned Stretch (Not Implemented Yet)**:
- Extended Extraction: Parse and store `date` (post/article date) and `rating` (e.g., star/score) at ingestion so the existing `date`/`rating` filters become fully effective across the corpus.

**Production Tradeoffs & Reflection:**

- If cost were no object, the preferred choice would be an embedding model with long context, multilingual support, domain fine-tuning for real estate specific terms, and low-latency hosted or on-prem inference (e.g., a hosted model such as `text-embedding-3-large`, or a model fine-tuned on housing & legal texts).
- Tradeoff Considerations: Accuracy vs. cost vs. latency vs. privacy (hosted API vs self-host). A local model such as mpnet wins on privacy & cost; a hosted model may win on raw accuracy & maintenance burden.
- Architecture Pattern: Cheap semantic index for recall + cross-encoder reranker for precision; cache embeddings and precompute reranks for popular queries.

**Evaluation & Tuning (Planned):**

- Experiment Grid: Chunk sizes (128, 256, 384) × `k_final` (3, 5, 8).
- Metrics: `recall@k`, `MRR`, final-answer accuracy, hallucination rate, and latency/cost per query.
- Current Verification is Manual: `scripts/query_documents.py ... --debug` prints the retrieved chunk text, distance, cosine similarity, and rerank score per query; `recall@k`/`MRR` are planned for the formal evaluation report. 

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System Prompt Grounding Instruction:**
- The system prompt enforces hard grounding by providing the LLM with a numbered set of retrieved context passages. It instructs the model repeatedly to answer ONLY from those passages, to never use outside knowledge or invent facts, and to reply with the refusal phrase "`I don't have enough information in the loaded documents to answer that.`" when passages do not provide sufficient context. 
- **System Prompt**: <br><br>"You are a grounded question-answering assistant for off-campus student housing near Penn State / State College, PA. You will be given a user question and a set of numbered context passages retrieved from a fixed corpus of forum posts, reviews, news articles, and official guides. You must obey ALL of the following rules:

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

**How Source Attribution is Surfaced in the Response:**
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

**Question That Failed:**
- `"Has Hendricks Investments been reported to enter units without notice?"`

**What the System Returned:**
- The system returned the refusal phrase, despite the fact that the corpus contains 2 adjacent sentences that state the claim cumulatively (one naming Hendricks, the next describing unauthorized entry). However, retrieval returned only one piece of the 2 fragments, so the LLM could not produce a grounded assetion.  

**Root cause (tied to a specific pipeline stage):**
- The chunking algorithm split a single factual claim across a boundary. As a result, the sentence tokenizer/HTML cleaning produced 2 chunks where the claim requires both. Retrieval returned a top-k that included only one of the fragments; the reranker scored fragments independently, so the LLM lacked the joined context. In summary, this is due to a combination of chunk boundary & candidate selection gap. 

**What you would change to fix it:**
- Increase overlap or adjust Greedy accumulation algorithm for medium/long documents so that sentence-spanning facts stay in at least one chunk. 
- Post-Retrieval: If top-k contains adjacent chunks from the same document, merge them (or include the nearest neighbor), before building the numbered context. 
- Add a lightweight BM25/phrase-match pass to the candidate pool to surface matching phrase neighbors, then rerank. 
- Add a unit test that constructs a synthetic document whose key fact is split across chunks, then assert that the system returns a grounded citation; used to verify the patch. 

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One Way the Spec Helped You During Implementation:**
- This spec enforced explicit grounding constraints up-front, including numbered context, canonical refusal methods, and temperature pinned to 0.
- This helped make design decisions significantly more straightforward & modular: short-circuit upon empty retrieval, build source lists programmatically, and verify end-to-end tests. 
- It also specified retrieval/rerank roles & metadata fields which let to a clear, classical RAG pipeline architecture: chunks -> embeddings -> retrieve -> prompt -> generate -> UI. 

**One Way Your Implementation Diverged From the Spec, and Why:**
- Implemented programmatic fallback: Cross-Encoder reranking is optional if sentence-transformers is not installed, with a tradeoff of accuracy. The embed/index defaults differ slightly between function defaults & higher-level configurations.
- This was implemented intentionally to keep the repository runnable in constrained development environments, and to support the higher quality `mpnet-base-v2` index for embedding. 

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

- *What I gave the AI:* A docstring template & all Python files in the repository (function signatures + desired style rules). I asked the model to generate comprehensive docstrings for every function: short summary, args, returns, raises, and a short example. This served a dual purpose: best practices in SWE/ML Engineering, and for conceptual explanation from an educational standpoint for unfamiliar library mechanics. 
- *What it produced:* Draft docstrings for functions across the codebase following the provided template.
- *What I changed or overrode:* I removed overly-robotic phrasing, standardized punctation (e.g., removing em-dashes), fixed capitalization, and corrected a few imprecise edge cases the model made. I harmonized wording across modules to a consistent voice. 

**Instance 2**

- *What I gave the AI:* The original `chunker.py` module implementation & the chunking spec from `planning.md`. I prompted the model via explaining the evident performance issue (multiple nested while loops, near-quadratic worst-case time complexity), and asked for a refractored Greedy algorithm chunk accumulation approach with optimized complexity. 
- *What it produced:* A refractored algorithm schema & code suggestions that flattened loop structure into a single-pass Greedy accumulator, plus notes on complexity, edge cases, and suggested unit checks (specified by the prompt for conceptual understanding).
- *What I changed or overrode:* Integrated the suggested algorithm into the repository. Renamed variables for clarity, preserved fallback behavior (sentence-first + hard floor), and defined a nested helper function for yielding chunks. Verified the optimized Greedy algorithm replaced the nested loops while keeping outputs as expected. 