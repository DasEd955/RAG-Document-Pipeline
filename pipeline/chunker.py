import re
import html
from typing import List, Dict, Optional, Iterable, Generator, Tuple, Any
from bs4 import BeautifulSoup
try:
    import tiktoken
except Exception:
    tiktoken = None

# _get_encoding_obj(): Returns a tiktoken "encoding" object used for tokenization (encode/decode), or none if unavailable
    # Tries to get the encoding based on the provided encoding name, or defaults to the encoding for "text-embedding-3-small" if no name is given 
    # Falls back to "cl100k_base" if that fails, and returns None if tiktoken is not available or all attempts fail
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
        # Edge Cases & Defaults: Use provided params or fall back to instance defaults; get tokenizer encoding object if possible
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
        if min_tokens is None:
            min_tokens = self.min_tokens
        enc = _get_encoding_obj(encoding_name or self.encoding_name)

        # Helper to construct a chunk dict with all metadata
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
        # Sentence Splitting & Token Counting: Split paragraph into sentences, get token counts and character spans for each sentence
        sentences = self.sentence_split(paragraph)
        if not sentences:
            return
        # Get token counts for each sentence using the tokenizer if available, otherwise approximate with word counts; compute character spans of sentences within the paragraph
        sent_toks = self._sentence_token_counts(sentences, enc)
        sent_spans = self._sentence_char_spans(paragraph, sentences)
        cumulative = [0]
        # Compute cumulative token counts to easily get token spans for sentence ranges; cumulative[i] gives the total tokens up to sentence i
        for t in sent_toks:
            cumulative.append(cumulative[-1] + t)

        n = len(sentences)
        i = 0
        idx = start_index
        # Greedy Sentence Accumulation: Iterate through sentences, accumulating them into chunks until reaching the token limit; handle long sentences and overlap as needed
        while i < n:
            j = i
            tok_sum = 0
            # Accumulate sentences until adding another would exceed the chunk_size token limit
            while j < n and tok_sum + sent_toks[j] <= chunk_size:
                tok_sum += sent_toks[j]
                j += 1
            # If no sentences fit (i.e. a single sentence exceeds chunk_size), we will handle that case separately by splitting the long sentence token-wise
            if j == i:
                s = sentences[i]
                s_start, s_end = sent_spans[i]
                # If we have a tokenizer encoding, split the long sentence into token-based chunks; otherwise, fall back to splitting by words
                if enc is not None:
                    tokens = enc.encode(s)
                    w = 0
                    # We will split the long sentence into chunks of chunk_size tokens, and compute the corresponding character spans for each chunk
                    while w < len(tokens):
                        slice_tokens = tokens[w: w + chunk_size]
                        chunk_text = enc.decode(slice_tokens)
                        tcount = len(slice_tokens)
                        token_start = cumulative[i] + w
                        token_end = token_start + tcount
                        pos_in_sentence = s.find(chunk_text)
                        # If the exact chunk text is not found in the sentence (which can happen due to tokenization quirks)
                        # We will approximate the character position by scaling based on the ratio of tokens to sentence length
                        if pos_in_sentence == -1:
                            pos_in_sentence = int(len(s) * w / max(1, len(tokens)))
                        char_start = s_start + pos_in_sentence
                        char_end = char_start + len(chunk_text)
                        # Yield the chunk with all metadata, then move to the next token slice
                        yield _make_chunk(chunk_text, token_start, token_end, char_start, char_end)
                        idx += 1
                        w += chunk_size
                # If we don't have a tokenizer encoding, we will split the long sentence by words, creating chunks of approximately chunk_size words
                # And compute character spans based on the position of the chunk text within the sentence
                else:
                    words = s.split()
                    w = 0
                    approx_tokens = len(words)
                    # If the sentence is very long in terms of words, we will create chunks of chunk_size words; otherwise, we will just yield the whole sentence as one chunk
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
                        # Yield the chunk with all metadata, then move to the next word slice
                        yield _make_chunk(chunk_text, token_start, token_end, char_start, char_end)
                        idx += 1
                        w += take
                # After handling the long sentence, we move to the next sentence    
                i += 1
                continue
            
            # For the normal case where we have accumulated one or more sentences that fit within the chunk_size token limit, we will create a chunk from sentences[i:j]
            chunk_text = " ".join(sentences[i:j])
            token_start = cumulative[i]
            token_end = cumulative[j]
            tok_count = token_end - token_start

            # If the accumulated sentences are fewer than min_tokens, we will try to add more sentences until we reach at least min_tokens or run out of sentences
            if tok_count < min_tokens and j < n:
                j += 1
                chunk_text = " ".join(sentences[i:j])
                token_end = cumulative[j]
                tok_count = token_end - token_start
            # If after trying to meet the min_tokens requirement we still have fewer tokens than min_tokens, we will just yield the chunk as is, since we don't want to create excessively large chunks
                # This allows for some flexibility in chunk sizes while still trying to meet the minimum token requirement
            char_start = sent_spans[i][0]
            char_end = sent_spans[j - 1][1]

            yield _make_chunk(chunk_text, token_start, token_end, char_start, char_end)
            idx += 1

            # Handle Overlap: If overlap is specified, we will move the starting index back by the appropriate number of sentences to create an overlap between chunks
                # We calculate how many sentences to go back based on the token counts of the sentences
            if overlap and overlap > 0:
                overlap_tokens = 0
                k = 0
                # We will move backwards from sentence j-1 to i, accumulating token counts until we reach the desired overlap token count, and set the next starting index accordingly
                for s_idx in range(j - 1, i - 1, -1):
                    overlap_tokens += sent_toks[s_idx]
                    k += 1
                    if overlap_tokens >= overlap:
                        break
                next_i = j - k if (j - k) > i else j
            # If no overlap is specified, we simply move to the next sentence after j
            else:
                next_i = j
            # Move to the next starting sentence index for the next chunk
            i = next_i
    
    # chunk_paragraph: Materialize & return the list of chunks for a single paragraph
    def chunk_paragraph(self, paragraph: str, doc_id: str, source: str,
                        section_heading: Optional[str], start_index: int = 0,
                        chunk_size: Optional[int] = None, overlap: Optional[int] = None,
                        min_tokens: Optional[int] = None, encoding_name: Optional[str] = None) -> List[Dict]:
        return list(self._iter_paragraph_chunks(paragraph, doc_id, source, section_heading, start_index,
                                                chunk_size, overlap, min_tokens, encoding_name))

    # chunk_text: Split text into paragraphs & produce chunks for each paragraph; supports streaming generator or full list
    def chunk_text(self, text: str, doc_id: str, source: str, section_heading: Optional[str] = None,
                   start_index: int = 0, chunk_size: Optional[int] = None, overlap: Optional[int] = None,
                   min_tokens: Optional[int] = None, encoding_name: Optional[str] = None,
                   stream: bool = False) -> Iterable[Dict]:
        paras = self.split_paragraphs(text)
        # Generator function to yield chunks for all paragraphs, maintaining a global chunk index across paragraphs
        def gen():
            idx = start_index
            for para in paras:
                for ch in self._iter_paragraph_chunks(para, doc_id, source, section_heading, idx,
                                                      chunk_size, overlap, min_tokens, encoding_name):
                    yield ch
                    idx += 1
        # Return a generator if streaming, otherwise materialize the full list of chunks
        if stream:
            return gen()
        return list(gen())

# When importing * from this module, only expose the Chunker class
__all__ = ["Chunker"]
