"""test_retrieval.py - Unit tests for Stage 4 hybrid-retrieval helpers.

These cover the pure, deterministic building blocks of hybrid search without
touching ChromaDB, sentence-transformers, or rank_bm25: the BM25 tokenizer,
Reciprocal Rank Fusion, the metadata filter predicate, and source boosting.

Run from project root: python -m pytest tests/test_retrieval.py -q
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import embeddings as emb


# ---- Tokenizer ---------------------------------------------------------------
def test_tokenizer_lowercases_and_splits_on_nonalphanumeric() -> None:
    """Verify the BM25 tokenizer lowercases and splits on non-alphanumeric runs.

    Entity names and numbers must survive as discrete lowercase tokens so that
    keyword recall can match them.
    """
    assert emb._tokenize_for_bm25("The Maxxen, 123 Beaver Ave.") == [
        "the", "maxxen", "123", "beaver", "ave",
    ]
    assert emb._tokenize_for_bm25("") == []


# ---- Reciprocal Rank Fusion --------------------------------------------------
def test_rrf_rewards_items_ranked_highly_by_both_retrievers() -> None:
    """Verify RRF scores an item appearing high in both rankings above either alone.

    'b' is rank 1 in the first list and rank 0 in the second, so its fused score
    must exceed that of items appearing in only one list.
    """
    fused = emb._rrf_fuse([["a", "b"], ["b", "c"]])
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def test_rrf_single_ranking_preserves_order() -> None:
    """Verify that with one ranking, RRF scores strictly decrease with rank."""
    fused = emb._rrf_fuse([["x", "y", "z"]])
    assert fused["x"] > fused["y"] > fused["z"]


# ---- Metadata Filtering ------------------------------------------------------
def test_no_filters_passes_everything() -> None:
    """Verify that a None or empty filter spec admits any chunk."""
    assert emb._passes_metadata_filter({"source": "anything"}, None) is True
    assert emb._passes_metadata_filter(None, {}) is True


def test_source_contains_is_case_insensitive_substring() -> None:
    """Verify source_contains matches as a case-insensitive substring."""
    meta = {"source": "documents/04_livingoffcampus.psu.edu_lease.html"}
    assert emb._passes_metadata_filter(meta, {"source_contains": "PSU.EDU"}) is True
    assert emb._passes_metadata_filter(meta, {"source_contains": "reddit.com"}) is False


def test_min_rating_and_missing_field_fails_closed() -> None:
    """Verify min_rating enforces a numeric threshold and rejects missing data.

    A chunk lacking the rating field cannot satisfy a rating constraint, so it
    must fail the filter rather than pass by default.
    """
    assert emb._passes_metadata_filter({"rating": 4.5}, {"min_rating": 4.0}) is True
    assert emb._passes_metadata_filter({"rating": 3.0}, {"min_rating": 4.0}) is False
    assert emb._passes_metadata_filter({"source": "x"}, {"min_rating": 4.0}) is False


def test_date_range_bounds_are_inclusive_and_require_field() -> None:
    """Verify date_after/date_before bound an ISO date field and require its presence."""
    meta = {"date": "2024-06-01"}
    assert emb._passes_metadata_filter(meta, {"date_after": "2024-01-01"}) is True
    assert emb._passes_metadata_filter(meta, {"date_before": "2024-01-01"}) is False
    assert emb._passes_metadata_filter({"source": "x"}, {"date_after": "2024-01-01"}) is False


# ---- Source Boosting ---------------------------------------------------------
def test_source_boost_scales_only_matching_sources() -> None:
    """Verify source boosting multiplies only the scores of matching sources.

    The boost mutates the fused score map in place; non-matching ids are untouched.
    """
    fused = {"a": 1.0, "b": 1.0}
    records = {
        "a": {"metadata": {"source": "documents/01_www.reddit.com_post.html"}},
        "b": {"metadata": {"source": "documents/04_livingoffcampus.psu.edu.html"}},
    }
    emb._apply_source_boost(fused, records, {"reddit.com": 2.0})
    assert fused["a"] == 2.0
    assert fused["b"] == 1.0


def test_source_boost_noop_when_none() -> None:
    """Verify that passing no boost map leaves fused scores unchanged."""
    fused = {"a": 1.0}
    emb._apply_source_boost(fused, {"a": {"metadata": {}}}, None)
    assert fused["a"] == 1.0
