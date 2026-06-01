# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

     The domain for elected pursuement regarding the RAG document system will be off-campus housing reviews & relevant safety information, such as reports of student apartments through forums like Reddit, CollegeConfidential, and/or Yelp reviews. 
     
     This knowledge is crucial & often hard to find due to the practical considerations of students seeking housing for the first time independently, along with the often vested interest of universities. In many institutions, such as the author's own Pennsylvania State University, housing demand far outnumbers available supply; the race to secure off-campus housing starts as early as a year prior. In addition, for many students, seeking off-campus housing may often be their first experience with factors such as leasing agreements, private property owners, and fiscal responsibility to a non-educational institution. 
     
     While it is the author's belief that most landlords are fair individuals who seek a return on their investment while providing a service to their communities, it is inevitable that in many large college towns, there exists some unscrupulous actors who fail to maintain their property, embed predatory/unenforceable clauses into the leasing agreement, or infringe on privacy such as entering without notifying a tenant. 
     
     In these situations, ignorance is often preyed upon by bad actors. Students who have only been exposed to educational institutions may falsely presume that private landlords will have their best interest at heart when signing a contract. Furthermore, in a world where even many older adults fail to read a contract carefully or have education in basic contract law, it should be no surprise that many students may fail to do so, signing predatory clauses. In addition, students may come from backgrounds where they are first-generation college students, first-time renters, or have no familial backing to help them navigate real estate contracts. 
     
     In addition, universities often have a vested financial interest, bar massive infringements, to avoid calling out these bad actors on official channels of communication. In many college towns, the institution is often one of the largest property portfolio holders. Publicly identifying bad landlords can result in devaluing surrounding real estate, spark controversy with influential donors, or hinder municipality relations. 
     
     The scope of the deficit is why access to this knowledge via unofficial forums of communication is crucial to student equity & fairness. As highlighted above, students who are from disadvantaged/non-historically represented backgrounds are often the most vulnerable to predatory practices in off-campus housing. It is the author's opinion that equalizing access to information is a key factor in driving equality. The purpose of this project shall be a case demo that tests the hypothesis that with access to information, students of all backgrounds will be able to make an educated choice in their best interests that allows them to secure safe, accessible housing from fair-practice property owners.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**

**Overlap:**

**Reasoning:**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.

2.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

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

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
