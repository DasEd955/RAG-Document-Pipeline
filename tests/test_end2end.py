"""
End-to-end tests for Stage 5 (generation + attribution).

These do NOT hit ChromaDB or the Groq API. We monkeypatch:
  - query.retrieve   -> deterministic fake retrieval results
  - query._generate  -> deterministic fake LLM output

That isolation lets us assert the two guarantees the pipeline promises,
independent of model behaviour or network:

  1. Grounding is enforced structurally: empty retrieval refuses WITHOUT
     calling the LLM, and a model refusal is detected and normalized.
  2. Source attribution is programmatic: the "sources" list is built from
     chunk metadata, not parsed from the model's text — so it is present even
     when the model output contains no citations, and absent on a refusal.

Run from the project root:  python -m pytest tests/test_end2end.py -q
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import query


# ---- fixtures / helpers ------------------------------------------------------
def _fake_results():
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


def _patch_retrieval(monkeypatch, results):
    monkeypatch.setattr(query, "retrieve", lambda *a, **k: {"query": "q", "results": results, "reranked": True})


def _patch_llm(monkeypatch, output):
    captured = {}

    def fake_generate(messages, llm_model, max_tokens, temperature):
        captured["messages"] = messages
        captured["llm_model"] = llm_model
        captured["temperature"] = temperature
        return output

    monkeypatch.setattr(query, "_generate", fake_generate)
    return captured


# ---- tests -------------------------------------------------------------------
def test_empty_retrieval_refuses_without_calling_llm(monkeypatch):
    _patch_retrieval(monkeypatch, [])

    def boom(*a, **k):
        raise AssertionError("LLM must not be called when there is no context")

    monkeypatch.setattr(query, "_generate", boom)

    res = query.ask("anything")
    assert res["grounded"] is False
    assert res["sources"] == []
    assert "couldn't find anything relevant" in res["answer"].lower()


def test_sources_are_programmatic_not_from_model(monkeypatch):
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


def test_model_refusal_is_detected_and_sources_suppressed(monkeypatch):
    _patch_retrieval(monkeypatch, _fake_results())
    # Model emits (a paraphrase of) the refusal phrase.
    _patch_llm(monkeypatch, query.REFUSAL_PHRASE)

    res = query.ask("What is the landlord's home phone number?")
    assert res["grounded"] is False
    assert res["sources"] == []
    assert res["answer"] == query.REFUSAL_PHRASE


def test_grounding_constraints_present_in_system_prompt(monkeypatch):
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


def test_blank_question_short_circuits(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("retrieve must not run for a blank question")

    monkeypatch.setattr(query, "retrieve", boom)
    res = query.ask("   ")
    assert res["grounded"] is False
    assert res["sources"] == []