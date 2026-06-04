"""test_end2end.py - End-to-end tests for Stage 5 (generation + source attribution).

Tests isolation strategy: monkeypatch retrieve() and _generate() to use
deterministic fake data (no ChromaDB or Groq API calls needed). Assertions
verify the two core grounding guarantees:

  1. Grounding is structurally enforced: empty retrieval refuses without calling
     the LLM; model refusals are detected and normalized.
  2. Source attribution is programmatic: the "sources" list is built from chunk
     metadata in ask(), not parsed from model output — so it persists even when
     the model omits citations, and is suppressed on refusal.

Run from project root: python -m pytest tests/test_end2end.py -q
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import query


# ---- Fixtures / Helpers ------------------------------------------------------
def _fake_results() -> list:
    """Return mock retrieval results for testing.

    Returns:
        list: A list of 2 fake result dicts with metadata, document text, and
              cosine similarity scores.
    """
    return [
        {
            "id": "doc1__0",
            "document": "Multiple students warn to avoid Hendricks Investments; "
                        "they report poor management and unreturned deposits.",
            "metadata": {
                "doc_id": "01_www.reddit.com_apartment_recommendations",
                "source": "documents\\01_www.reddit.com_apartment_recommendations.html",
                "chunk_index": 0,
            },
            "cosine_similarity": 0.81,
        },
        {
            "id": "doc2__3",
            "document": "Hendricks was a nightmare to deal with for maintenance requests.",
            "metadata": {
                "doc_id": "09_www.reddit.com_off_campus_housing",
                "source": "documents\\09_www.reddit.com_off_campus_housing.html",
                "chunk_index": 3,
            },
            "cosine_similarity": 0.74,
        },
    ]


def _patch_retrieval(monkeypatch, results: list) -> None:
    """Monkeypatch query.retrieve to return deterministic results.

    Args:
        monkeypatch: pytest fixture for patching.
        results (list): Fake results to return from retrieve().
    """
    monkeypatch.setattr(query, "retrieve", lambda *a, **k: {"query": "q", "results": results, "reranked": True})


def _patch_llm(monkeypatch, output: str) -> dict:
    """Monkeypatch query._generate to return a fixed string and capture call args.

    Args:
        monkeypatch: pytest fixture for patching.
        output (str): The fake LLM response to return.

    Returns:
        dict: A dict that will be populated with captured call arguments:
              messages, llm_model, temperature.
    """
    captured = {}

    def fake_generate(messages, llm_model, max_tokens, temperature):
        captured["messages"] = messages
        captured["llm_model"] = llm_model
        captured["temperature"] = temperature
        return output

    monkeypatch.setattr(query, "_generate", fake_generate)
    return captured


# ---- Tests -------------------------------------------------------------------
def test_empty_retrieval_refuses_without_calling_llm(monkeypatch) -> None:
    """Verify that empty retrieval refuses without calling the LLM.

    If ChromaDB returns no results, ask() must return the refusal phrase
    immediately without invoking the generation model. This prevents the
    model from hallucinating an answer when there is no grounding context.
    """
    _patch_retrieval(monkeypatch, [])

    def boom(*a, **k):
        raise AssertionError("LLM must not be called when there is no context")

    monkeypatch.setattr(query, "_generate", boom)

    res = query.ask("anything")
    assert res["grounded"] is False
    assert res["sources"] == []
    assert "couldn't find anything relevant" in res["answer"].lower()


def test_sources_are_programmatic_not_from_model(monkeypatch) -> None:
    """Verify that sources are built from metadata, not parsed from model output.

    Even if the mocked LLM emits zero citations, result["sources"] must still
    contain the full source list, built purely from chunk metadata. This proves
    source attribution is guaranteed programmatically, not delegated to the LLM.
    """
    # Model output contains NO citations at all...
    _patch_retrieval(monkeypatch, _fake_results())
    _patch_llm(monkeypatch, "Students strongly advise avoiding Hendricks Investments.")

    res = query.ask("What do students say about Hendricks Investments?")

    assert res["grounded"] is True
    # ...yet sources are still present, built from chunk metadata.
    assert len(res["sources"]) == 2
    # Programmatic resolution mapped the ordinal prefix to the real URL.
    assert any("reddit.com" in s for s in res["sources"])
    # And each source carries its [n] marker aligned with the context block.
    assert res["sources"][0].startswith("[1]")
    assert res["sources"][1].startswith("[2]")


def test_model_refusal_is_detected_and_sources_suppressed(monkeypatch) -> None:
    """Verify that model refusals are detected and sources are suppressed.

    If the LLM emits the canonical refusal phrase, ask() must return
    grounded=False and an empty sources list. There is no grounded claim to
    attribute, so attribution would be misleading.
    """
    _patch_retrieval(monkeypatch, _fake_results())
    # Model emits (a paraphrase of) the refusal phrase.
    _patch_llm(monkeypatch, query.REFUSAL_PHRASE)

    res = query.ask("What is the landlord's home phone number?")
    assert res["grounded"] is False
    assert res["sources"] == []
    assert res["answer"] == query.REFUSAL_PHRASE


def test_grounding_constraints_present_in_system_prompt(monkeypatch) -> None:
    """Verify that the system prompt enforces grounding constraints.

    The system prompt must contain hard language (ONLY, must NOT), the refusal
    phrase, numbered context, and temperature must be pinned to 0 for determinism.
    """
    _patch_retrieval(monkeypatch, _fake_results())
    captured = _patch_llm(monkeypatch, "Downtown is reportedly expensive [1].")

    query.ask("Is downtown expensive?")

    system = captured["messages"][0]["content"]
    assert captured["messages"][0]["role"] == "system"
    # Hard grounding language, the refusal phrase, and the numbered context.
    assert "ONLY" in system
    assert query.REFUSAL_PHRASE in system
    assert "[1]" in system and "[2]" in system
    # Temperature pinned to 0 for determinism / grounding.
    assert captured["temperature"] == 0.0
    # Generation model is the Groq llama model from spec.
    assert captured["llm_model"] == "llama-3.3-70b-versatile"


def test_blank_question_short_circuits(monkeypatch) -> None:
    """Verify that blank/whitespace-only questions are rejected early.

    ask() must validate input and refuse to call retrieve() or generate()
    on empty questions. This is a basic input sanity check.
    """
    def boom(*a, **k):
        raise AssertionError("retrieve must not run for a blank question")

    monkeypatch.setattr(query, "retrieve", boom)
    res = query.ask("   ")
    assert res["grounded"] is False
    assert res["sources"] == []