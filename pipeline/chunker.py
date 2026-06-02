import re
import html
from typing import List, Dict, Optional, Iterable, Generator, Tuple, Any
from bs4 import BeautifulSoup
try:
    import tiktoken
except Exception:
    tiktoken = None

# _get_encoding_obj(): Returns a tiktoken "encoding" object used for tokenization (encode/decode), or none if unavailable
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

# _token_count_with_enc(): Returns the token count for text using a provided tokenizer enc
    # Falls back to cheap word-count if enc is unavailable or fails
def _token_count_with_enc(text: str, enc: Any) -> int:
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text.split())

# Chunker Class: Token-aware, sentence-first text chunker that yields chunks with token/char spans & provenance
    # Orchestrates cleaning, paragraph & sentence splitting, and generation of chunks suitable for embedding/indexing
class Chunker:
    # Initialize chunking parameters & optional tokenizer encoding name
    def __init__(self, chunk_size: int = 256, overlap: int = 64, min_tokens: int = 50,
                 encoding_name: Optional[str] = None):
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.min_tokens = int(min_tokens)
        self.encoding_name = encoding_name

    # clean_html(): Strips HTML boilerplate & normalizes markup to clean plain-text
        # Uses BeautifulSoup to remove scripts/styles/nav etc., extracts text, unescapes HTML entities
        # Normalizes line endings/whitspace & returns trimmed text ready for splitting
    def clean_html(self, raw: str) -> str:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = html.unescape(text)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n\s+\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    # split_paragraphs(): Split normalized text into paragraphs on consecutive line blanks
    def split_paragraphs(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return paras

    # sentence_split(): Split normalized text into sentences using punctuation & capitalization
        # Note: Handles most cases, but is not a full NLP sentence parser
    def sentence_split(self, paragraph: str) -> List[str]:
        pieces = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(])', paragraph)
        return [p.strip() for p in pieces if p.strip()]

    # _sentence_token_counts(): Return token counts per sentence using a tokenizer if available
        # If a tiktoken encoding is available gives exact BPE counts, otherwise falls back to an approximate word-based count
    def _sentence_token_counts(self, sentences: List[str], enc: Any = None) -> List[int]:
        return [_token_count_with_enc(s, enc) for s in sentences]

    # _sentence_char_spans(): Compute (start, end) character spans of each sentence inside the paragraph
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
    
    # Yield token-aware chunks for a paragraph by greedily accumulating whole sentences
        # Splits long sentences token-wise, produces token_span & char_span, and supports overlap
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
                        yield {
                            "doc_id": doc_id,
                            "source": source,
                            "section_heading": section_heading,
                            "chunk_index": idx,
                            "text": chunk_text,
                            "token_count": tcount,
                            "token_span": [token_start, token_end],
                            "char_span": [char_start, char_end],
                        }
                        idx += 1
                        w += chunk_size
                else:
                    words = s.split()
                    w = 0
                    approx_tokens = len(words)
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
                        yield {
                            "doc_id": doc_id,
                            "source": source,
                            "section_heading": section_heading,
                            "chunk_index": idx,
                            "text": chunk_text,
                            "token_count": tcount,
                            "token_span": [token_start, token_end],
                            "char_span": [char_start, char_end],
                        }
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

            yield {
                "doc_id": doc_id,
                "source": source,
                "section_heading": section_heading,
                "chunk_index": idx,
                "text": chunk_text,
                "token_count": tok_count,
                "token_span": [token_start, token_end],
                "char_span": [char_start, char_end],
            }
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
    
    # Materialize & return the list of chunks for a single paragraph
    def chunk_paragraph(self, paragraph: str, doc_id: str, source: str,
                        section_heading: Optional[str], start_index: int = 0,
                        chunk_size: Optional[int] = None, overlap: Optional[int] = None,
                        min_tokens: Optional[int] = None, encoding_name: Optional[str] = None) -> List[Dict]:
        return list(self._iter_paragraph_chunks(paragraph, doc_id, source, section_heading, start_index,
                                                chunk_size, overlap, min_tokens, encoding_name))

    # Split text into paragraphs & produce chunks for each paragraph; supports streaming generator or full list
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
