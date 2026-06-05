"""One-off generator for RAG-Architecture.png. Reflects the real tech stack.
Run: python _gen_diagram.py from the repo root. Requires Pillow (PIL) and the specified fonts.
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 1290
BG = (23, 23, 26)
INK = (235, 236, 238)
SUB = (188, 192, 200)
ARROW = (150, 154, 162)

F = "C:/Windows/Fonts/arial.ttf"
FB = "C:/Windows/Fonts/arialbd.ttf"
FM = "C:/Windows/Fonts/consola.ttf"

def font(path, size):
    return ImageFont.truetype(path, size)

f_title = font(FB, 32)
f_sub = font(F, 17)
f_stage = font(FB, 23)
f_detail = font(F, 15)
f_mono = font(FM, 13)
f_small = font(F, 13)
f_smallb = font(FB, 14)
f_label = font(FB, 14)
f_vert = font(FB, 15)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def center(draw, cx, y, text, fnt, fill):
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)


def box(x, y, w, h, fill, title=None, lines=None, *, title_font=f_stage,
        line_font=f_detail, title_fill=INK, line_fill=(225, 228, 234),
        radius=14, align_center=True, sub=None):
    d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill)
    cx = x + w / 2
    cy = y + 14
    if title:
        if align_center:
            center(d, cx, cy, title, title_font, title_fill)
        else:
            d.text((x + 16, cy), title, font=title_font, fill=title_fill)
        cy += title_font.size + 8
    if sub:
        if align_center:
            center(d, cx, cy, sub, f_small, SUB)
        else:
            d.text((x + 16, cy), sub, font=f_small, fill=SUB)
        cy += f_small.size + 8
    for ln in (lines or []):
        if align_center:
            center(d, cx, cy, ln, line_font, line_fill)
        else:
            d.text((x + 16, cy), ln, font=line_font, fill=line_fill)
        cy += line_font.size + 5
    return (x, y, x + w, y + h)


def varrow(cx, y0, y1, label=None):
    d.line([cx, y0, cx, y1], fill=ARROW, width=3)
    # arrowhead
    d.polygon([(cx - 7, y1 - 10), (cx + 7, y1 - 10), (cx, y1)], fill=ARROW)
    if label:
        w = d.textlength(label, font=f_label)
        pad = 7
        ly = (y0 + y1) / 2 - f_label.size / 2
        d.rectangle([cx + 14, ly - pad + 2, cx + 14 + w + 2 * pad, ly + f_label.size + pad - 2], fill=BG)
        d.text((cx + 14 + pad, ly), label, font=f_label, fill=(210, 213, 220))


# palette
C_IN = (38, 110, 74)      # green  - sources
C1 = (31, 111, 92)        # teal   - ingestion
C2 = (74, 62, 140)        # indigo - chunking
C3 = (33, 86, 166)        # blue   - embedding
C4 = (122, 102, 36)       # gold   - retrieval
C5 = (138, 52, 40)        # red    - generation
C_OUT = (70, 72, 80)      # gray   - output
C_SIDE_L = (58, 53, 86)   # left annotations
C_SIDE_R = (96, 80, 28)   # right annotations
C_QUERY = (40, 96, 70)    # user query

CX = 520            # center x of main column
BW = 560            # main box width
LX = CX - BW / 2    # left edge of main column

# ---- Title -------------------------------------------------------------------
center(d, W / 2, 26, "RAG Document Pipeline", f_title, INK)
center(d, W / 2, 66, "Unofficial Off-Campus Housing Guide  -  Penn State / State College, PA", f_sub, SUB)

# ---- Sources (input) ---------------------------------------------------------
y = 108
box(LX, y, BW, 70, C_IN,
    title="Sources  -  12 HTML documents",
    lines=["Reddit . TripAdvisor . Penn State guide . College Magazine . Daily Collegian . Facebook"],
    title_font=f_stage, line_font=f_small)
varrow(CX, y + 70, y + 70 + 36, "raw HTML")

# ---- Stage 1: Ingestion ------------------------------------------------------
y = 250
box(LX, y, BW, 118, C1,
    title="Stage 1 . Document Ingestion",
    sub="scripts/download_documents.py",
    lines=["requests (HTTP, retries)  ->  Playwright (Chromium) JS fallback",
           "URLs parsed from planning.md . polite retry/backoff",
           "saves raw .html + download_metadata.json (url <-> file)"])
varrow(CX, y + 118, y + 118 + 40, "raw HTML / text")

# ---- Stage 2: Chunking -------------------------------------------------------
y = 408
s2 = box(LX, y, BW, 150, C2,
    title="Stage 2 . Preprocessing & Chunking",
    sub="pipeline/chunker.py  .  pipeline/ingest.py",
    lines=["BeautifulSoup clean + inline boilerplate scrub",
           "tiktoken (cl100k_base) token counts . regex sentence split",
           "sentence-first greedy; adaptive by length:",
           ">4k -> 256/64   .   1.5-4k -> 192/48   .   <1.5k -> 128/32",
           "50-token hard floor . content-hash dedup"])
# left annotation for stage 2 metadata
box(20, y + 6, 230, 138, C_SIDE_L,
    title="Chunk metadata", title_font=f_smallb,
    lines=["doc_id . source", "chunk_index", "char_span . token_span", "token_count",
           "", "-> chunks/chunks.jsonl"],
    line_font=f_small, align_center=False)
varrow(CX, y + 150, y + 150 + 40, "clean chunks + metadata (JSONL)")

# ---- Stage 3: Embedding + Vector Store --------------------------------------
y = 598
box(LX, y, BW, 122, C3,
    title="Stage 3 . Embedding + Vector Store",
    sub="pipeline/embeddings.py",
    lines=["sentence-transformers  all-mpnet-base-v2  (768-d, L2-normalized)",
           "ChromaDB PersistentClient . hnsw:space = cosine",
           "model-consistency guard (embedding_model in metadata)"])
varrow(CX, y + 122, y + 122 + 40, "cosine top-20 candidates")

# ---- Stage 4: Retrieval (Hybrid) --------------------------------------------
y = 760
box(LX, y, BW, 158, C4,
    title="Stage 4 . Retrieval  (Hybrid)",
    sub="pipeline/embeddings.py",
    lines=["Semantic: cosine recall, k_candidate = 20",
           "Keyword: BM25 (rank_bm25) top-20  ->  exact entity match",
           "Reciprocal Rank Fusion  (RRF, k = 60)",
           "Metadata filter (source/date/rating) + source boost",
           "Cross-encoder rerank: ms-marco-MiniLM-L-6-v2  ->  top-k (3-5)"])
# left annotation: filter/boost
box(20, y + 18, 230, 96, C_SIDE_L,
    title="Filter / boost by", title_font=f_smallb,
    lines=["source  (active)", "date  (plumbed)", "rating  (plumbed)"],
    line_font=f_small, align_center=False)
# right annotation: user query + conversational memory
box(770, y + 6, 250, 150, C_QUERY,
    title="User query", title_font=f_smallb,
    lines=["CLI . Gradio chat UI", "",
           "Conversational memory:", "condense follow-up", "-> standalone query",
           "(app/conversation.py)"],
    line_font=f_small, align_center=False)
# connector from query box to stage 4
d.line([770, y + 80, CX + BW / 2, y + 80], fill=ARROW, width=3)
d.polygon([(CX + BW / 2 + 10, y + 80 - 6), (CX + BW / 2 + 10, y + 80 + 6), (CX + BW / 2, y + 80)], fill=ARROW)
varrow(CX, y + 158, y + 158 + 40, "top-k chunks + source refs")

# ---- Stage 5: Generation -----------------------------------------------------
y = 958
box(LX, y, BW, 138, C5,
    title="Stage 5 . Grounded Generation",
    sub="app/query.py",
    lines=["Groq  llama-3.3-70b-versatile  .  temperature = 0",
           "Answer ONLY from passages; refuse if insufficient",
           "Programmatic [n] -> source URL attribution",
           "Multi-turn: history-aware prompt (app/conversation.py)"])
varrow(CX, y + 138, y + 138 + 40, "cited answer")

# ---- Output ------------------------------------------------------------------
y = 1136
box(LX, y, BW, 80, C_OUT,
    title="Cited answer to user",
    lines=["Gradio chat UI (gr.Chatbot + gr.State)  .  CLI  .  every claim cites source chunk(s)"],
    title_font=f_stage, line_font=f_small)

# ---- Right-side vertical cross-cutting annotation ---------------------------
# Tests + config + env, drawn as a tall side note on the far right.
side_x = 1045
box(side_x, 250, 215, 846, (44, 44, 50),
    title="Cross-cutting", title_font=f_smallb,
    lines=["Tests: pytest (19)", "  test_retrieval", "  test_end2end", "  test_conversation",
           "", "Config: python-dotenv", "  GROQ_API_KEY (.env)",
           "", "Graceful degradation:", "  no rank_bm25 -> semantic", "  no cross-encoder", "     -> RRF order",
           "", "Caching (per proc):", "  models . corpus . BM25"],
    line_font=f_small, align_center=False)

# ---- Footer ------------------------------------------------------------------
center(d, W / 2, 1245,
       "Flow: Ingest -> Chunk -> Embed/Index -> Hybrid Retrieve (semantic + BM25 -> RRF -> filter/boost -> rerank) -> Grounded multi-turn generation",
       f_small, SUB)

img.save("RAG-Architecture.png")
print("wrote RAG-Architecture.png", img.size)
