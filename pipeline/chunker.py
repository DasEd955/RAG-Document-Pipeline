import re
import html
from typing import List, Dict, Optional, Iterable, Generator, Tuple, Any
from bs4 import BeautifulSoup
try:
    import tiktoken
except Exception:
    tiktoken = None


def _get_encoding_obj(encoding_name: Optional[str] = None) -> Any:
    if tiktoken is None:
        return None
    try:
        if encoding_name:
            return tiktoken.get_encoding(encoding_name)
        return tiktoken.encoding_for_model("text-embedding-3-small")
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _token_count_with_enc(text: str, enc: Any) -> int:
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text.split())


# ---- inline boilerplate scrubbing -------------------------------------------
# These run on the EXTRACTED text (after BeautifulSoup), because the worst
# offenders -- share toolbars, newsletter blurbs, reaction counts -- get
# flattened into the same text block as the article body, so whole-line
# filtering can't isolate them. Each pattern maps to a concrete artifact seen
# in real output, and every removal is conservative to avoid eating content.

_SHARE_WORDS = r"Facebook|Twitter|WhatsApp|SMS|Email|Print|Save|LinkedIn|Reddit|Pinterest|Flipboard|Copy Link"
# Only fires on 3+ consecutive share words, so a lone "Facebook" in a sentence survives.
_SHARE_RUN = re.compile(rf"(?:\b(?:{_SHARE_WORDS})\b[\s|·]*){{3,}}", re.I)

_INLINE_PHRASE_PATTERNS = [
    re.compile(r"Daily Collegian Newsletter.*?Sign up here!?", re.I | re.S),
    re.compile(r"Summarized by AI from the post below", re.I),
    re.compile(r"All reactions:?\s*\d*", re.I),
    re.compile(r"\bSee more\b", re.I),
    re.compile(r"\bSign up here!?", re.I),
    re.compile(r"Send Letter to the Editor", re.I),
    re.compile(r"(?:Author\s+(?:linkedin|twitter|email|facebook)\s*){2,}", re.I),
    re.compile(r"\bOther posts\b", re.I),
    # relative timestamps: "41w", "1y", "1y ago", "3 mo ago", "2 days ago"
    re.compile(r"\b\d+\s*[wy]\b(?:\s+ago)?", re.I),
    re.compile(r"\b\d+\s*(?:mo|hr|min)\b\s+ago\b", re.I),
    re.compile(r"\b\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago\b", re.I),
]

# Everything from the earliest of these markers onward is end-of-article
# navigation / related links / tag clusters, so drop the tail.
_TAIL_MARKERS = [
    re.compile(r"MORE NEWS CONTENT", re.I),
    re.compile(r"Submit a Letter to the Editor", re.I),
]

# Collapse a phrase (15+ chars) that is immediately repeated back-to-back.
_DUP_PHRASE = re.compile(r"(\b[\w][\w ,'&.\-]{15,}?)(?:\s+\1\b)+", re.I)


class Chunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 128, min_tokens: int = 100,
                 encoding_name: Optional[str] = None):
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.min_tokens = int(min_tokens)
        self.encoding_name = encoding_name

    # _scrub_inline_boilerplate(): remove boilerplate that survived into the body text
    def _scrub_inline_boilerplate(self, text: str) -> str:
        # 1) Cut trailing navigation at the earliest end-of-article marker.
        cut = len(text)
        for m in _TAIL_MARKERS:
            hit = m.search(text)
            if hit:
                cut = min(cut, hit.start())
        text = text[:cut]
        # 2) Remove social share-button runs.
        text = _SHARE_RUN.sub(" ", text)
        # 3) Remove known boilerplate phrases.
        for pat in _INLINE_PHRASE_PATTERNS:
            text = pat.sub(" ", text)
        # 4) Collapse immediately-repeated phrases (FB spam, repeated nav lines).
        for _ in range(2):
            text = _DUP_PHRASE.sub(r"\1", text)
        # 5) Tidy whitespace left behind by removals.
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    def clean_html(self, raw: str) -> str:
        """Strip HTML boilerplate & normalize markup to clean plain-text."""
        soup = BeautifulSoup(raw, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript",
                         "form", "aside", "iframe", "button", "svg", "figure", "menu"]):
            tag.decompose()

        def _extract_main_text(s: BeautifulSoup) -> str:
            candidates = []
            for tname in ("main", "article"):
                for el in s.find_all(tname):
                    txt = el.get_text(" \n", strip=True)
                    if txt:
                        candidates.append((len(txt), txt))
            role_main = s.find(attrs={"role": "main"})
            if role_main:
                txt = role_main.get_text(" \n", strip=True)
                if txt:
                    candidates.append((len(txt), txt))
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]
            div_cands = []
            for el in s.find_all(["div", "section"]):
                txt = el.get_text(" \n", strip=True)
                if txt and len(txt) > 200:
                    div_cands.append((len(txt), txt))
            if div_cands:
                div_cands.sort(reverse=True)
                return div_cands[0][1]
            return s.get_text(separator="\n")

        text = _extract_main_text(soup)
        text = html.unescape(text)

        # Normalize whitespace, non-breaking spaces, stray replacement chars.
        text = text.replace("\u00a0", " ")
        text = text.replace("\ufffd", " ")
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        # NEW: scrub inline boilerplate that got flattened into the body text.
        text = self._scrub_inline_boilerplate(text)

        lines = [l.strip() for l in text.splitlines()]
        lines = [l for l in lines if l]

        boilerplate_line_patterns = [
            r"^read\s*more\b",
            r"^continue\s+reading\b",
            r"^share\b",
            r"^people\s+also\s+ask\b",
            r"^top\s+.*for\b",
            r"^search\b",
            r"^by\s+continuing,",
            r"user\s+agreement",
            r"privacy\s+policy",
            r"all\s+rights\s+reserved",
            r"continue\s+with",
            r"^reddit,?\s+inc\b",
            r"^reddit\s+rules\b",
            r"^your\s+privacy\s+choices\b",
            r"^accessibility\b",
            r"^continue\s+with\s+phone\s+number\b",
            r"^continue\s+with\s+email\b",
            r"^search\s+real\s+takes\b",
            r"^and\s+acknowledge\s+that\s+you\s+understand\s+the\b",
            r"^home$",
            r"^popular$",
            r"^news$",
            r"^explore$",
            r"^best\s+of\b",
            r"^advert",
            r"^sponsored",
            r"^cookie",
            r"^subscribe",
            r"^\d+\s?(mo|hour|hours|day|days|yesterday|today)\b",
            r"^skip\s*to\s*main\b",
            r"^top\s+posts\b",
            r"^top\s+posts\s+of\b",
            r"^sign\s*in\b",
            r"^sign\s*in\s*to\b",
            r"^last\s+updated\b",
            r"^more\s+like\s+this\b",
            r"^browse\s+all\b",
            r"^browse\s+forums\b",
            r"^reddit\s*-\s*the\s*heart\b",
            r"^inbox\b",
            r"^see\s+all\b",
            r"^comments?\b",
            # Additional patterns for the corpus
            r"^tags\b",
            r"^in\s+this\s+series\b",
            r"^previous\s+next\b",
            r"^\d+\s+updates?\b",
            r"^join\b",
        ]
        compiled = [re.compile(p, flags=re.I) for p in boilerplate_line_patterns]

        def is_boilerplate_line(l: str) -> bool:
            if not l:
                return True
            low = l.lower()
            for p in compiled:
                if p.search(low):
                    return True
            if re.match(r"^https?://\S+$", low):
                return True
            if re.search(r"\b(skip to main|skip to content|top posts|sign in|last updated|more like this|browse all|browse forums|reddit - the heart|inbox|see all|comments?)\b", low):
                return True
            if "|" in l and any(k in low for k in ("skip to content", "resources", "off-campus", "off campus", "penn state", "search", "home")):
                return True
            if len(low.split()) <= 2 and len(low) <= 15 and low in ("home", "popular", "news", "explore", "search", "public", "menu", "more"):
                return True
            if re.match(r"^[^A-Za-z0-9]+$", l):
                return True
            return False

        filtered = [l for l in lines if not is_boilerplate_line(l)]

        from collections import Counter
        counts = Counter(filtered)
        filtered = [l for l in filtered if counts[l] <= 3]

        paragraphs = []
        cur = []
        for l in filtered:
            if not l:
                if cur:
                    paragraphs.append(" ".join(cur).strip())
                    cur = []
            else:
                cur.append(l)
        if cur:
            paragraphs.append(" ".join(cur).strip())

        cleaned_paras = []
        for p in paragraphs:
            words = p.split()
            if len(words) < 5:
                continue
            alpha_chars = sum(1 for ch in p if ch.isalpha())
            alpha_ratio = alpha_chars / max(1, len(p))
            if len(p) < 150 and alpha_ratio < 0.55:
                continue
            cleaned_paras.append(p)

        result = "\n\n".join([p for p in cleaned_paras if p])
        result = re.sub(r"\n\s+\n", "\n\n", result)
        result = re.sub(r"[ \t]+", " ", result)
        return result.strip()

    def split_paragraphs(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return paras

    def sentence_split(self, paragraph: str) -> List[str]:
        pieces = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(])', paragraph)
        return [p.strip() for p in pieces if p.strip()]

    def _sentence_token_counts(self, sentences: List[str], enc: Any = None) -> List[int]:
        return [_token_count_with_enc(s, enc) for s in sentences]

    def _sentence_char_spans(self, paragraph: str, sentences: List[str]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        pos = 0
        for s in sentences:
            idx = paragraph.find(s, pos)
            if idx == -1:
                idx = paragraph.find(s.strip(), pos)
            if idx == -1:
                idx = pos
            start = idx
            end = start + len(s)
            spans.append((start, end))
            pos = end
        return spans

    def _iter_paragraph_chunks(self, paragraph: str, doc_id: str, source: str,
                               section_heading: Optional[str], start_index: int = 0,
                               chunk_size: Optional[int] = None, overlap: Optional[int] = None,
                               min_tokens: Optional[int] = None, encoding_name: Optional[str] = None) -> Generator[Dict, None, None]:
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if min_tokens is None:
            min_tokens = self.min_tokens
        enc = _get_encoding_obj(encoding_name or self.encoding_name)

        def _make_chunk(chunk_text: str, token_start: int, token_end: int, char_start: int, char_end: int) -> Dict:
            return {
                "doc_id": doc_id,
                "source": source,
                "section_heading": section_heading,
                "chunk_index": idx,
                "text": chunk_text,
                "token_count": token_end - token_start,
                "token_span": [token_start, token_end],
                "char_span": [char_start, char_end],
            }

        sentences = self.sentence_split(paragraph)
        if not sentences:
            return
        sent_toks = self._sentence_token_counts(sentences, enc)
        sent_spans = self._sentence_char_spans(paragraph, sentences)
        cumulative = [0]
        for t in sent_toks:
            cumulative.append(cumulative[-1] + t)

        n = len(sentences)
        i = 0
        idx = start_index
        while i < n:
            j = i
            tok_sum = 0
            while j < n and tok_sum + sent_toks[j] <= chunk_size:
                tok_sum += sent_toks[j]
                j += 1
            if j == i:
                s = sentences[i]
                s_start, s_end = sent_spans[i]
                if enc is not None:
                    tokens = enc.encode(s)
                    w = 0
                    while w < len(tokens):
                        slice_tokens = tokens[w: w + chunk_size]
                        chunk_text = enc.decode(slice_tokens)
                        tcount = len(slice_tokens)
                        token_start = cumulative[i] + w
                        token_end = token_start + tcount
                        pos_in_sentence = s.find(chunk_text)
                        if pos_in_sentence == -1:
                            pos_in_sentence = int(len(s) * w / max(1, len(tokens)))
                        char_start = s_start + pos_in_sentence
                        char_end = char_start + len(chunk_text)
                        yield _make_chunk(chunk_text, token_start, token_end, char_start, char_end)
                        idx += 1
                        w += chunk_size
                else:
                    words = s.split()
                    w = 0
                    while w < len(words):
                        take = min(len(words) - w, chunk_size)
                        chunk_words = words[w: w + take]
                        chunk_text = " ".join(chunk_words)
                        tcount = len(chunk_words)
                        token_start = cumulative[i] + w
                        token_end = token_start + tcount
                        pos_in_sentence = s.find(chunk_text)
                        if pos_in_sentence == -1:
                            pos_in_sentence = 0
                        char_start = s_start + pos_in_sentence
                        char_end = char_start + len(chunk_text)
                        yield _make_chunk(chunk_text, token_start, token_end, char_start, char_end)
                        idx += 1
                        w += take
                i += 1
                continue

            chunk_text = " ".join(sentences[i:j])
            token_start = cumulative[i]
            token_end = cumulative[j]
            tok_count = token_end - token_start

            if tok_count < min_tokens and j < n:
                j += 1
                chunk_text = " ".join(sentences[i:j])
                token_end = cumulative[j]
                tok_count = token_end - token_start
            char_start = sent_spans[i][0]
            char_end = sent_spans[j - 1][1]

            yield _make_chunk(chunk_text, token_start, token_end, char_start, char_end)
            idx += 1

            if overlap and overlap > 0:
                overlap_tokens = 0
                k = 0
                for s_idx in range(j - 1, i - 1, -1):
                    overlap_tokens += sent_toks[s_idx]
                    k += 1
                    if overlap_tokens >= overlap:
                        break
                next_i = j - k if (j - k) > i else j
            else:
                next_i = j
            i = next_i

    def chunk_paragraph(self, paragraph: str, doc_id: str, source: str,
                        section_heading: Optional[str], start_index: int = 0,
                        chunk_size: Optional[int] = None, overlap: Optional[int] = None,
                        min_tokens: Optional[int] = None, encoding_name: Optional[str] = None) -> List[Dict]:
        return list(self._iter_paragraph_chunks(paragraph, doc_id, source, section_heading, start_index,
                                                chunk_size, overlap, min_tokens, encoding_name))

    def chunk_text(self, text: str, doc_id: str, source: str, section_heading: Optional[str] = None,
                   start_index: int = 0, chunk_size: Optional[int] = None, overlap: Optional[int] = None,
                   min_tokens: Optional[int] = None, encoding_name: Optional[str] = None,
                   stream: bool = False) -> Iterable[Dict]:
        paras = self.split_paragraphs(text)

        def gen():
            idx = start_index
            for para in paras:
                for ch in self._iter_paragraph_chunks(para, doc_id, source, section_heading, idx,
                                                      chunk_size, overlap, min_tokens, encoding_name):
                    yield ch
                    idx += 1
        if stream:
            return gen()
        return list(gen())


__all__ = ["Chunker"]