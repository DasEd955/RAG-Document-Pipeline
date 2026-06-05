"""test_conversation.py - Tests for conversational memory (multi-turn Q&A).

Isolation strategy: monkeypatch query.retrieve, query._generate, and
query.condense_question with deterministic fakes (no ChromaDB or Groq calls).
Assertions verify that memory changes retrieval and generation as intended while
the grounding contract still holds:

  1. A follow-up is condensed into a standalone query using prior turns, and that
     condensed query (not the raw follow-up) is what retrieval sees.
  2. Prior turns are injected into the generation prompt for reference resolution.
  3. Grounding is preserved: empty retrieval still refuses.
  4. The Conversation object accumulates turns and forwards a bounded window.

Run from project root: python -m pytest tests/test_conversation.py -q
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import query
from app.conversation import Conversation


# ---- Helpers -----------------------------------------------------------------
def _fake_results() -> list:
    """Return a single mock retrieval result.

    Returns:
        list: A one-element list with a fake chunk record.
    """
    return [{
        "id": "doc1__0",
        "document": "The Maxxen is a luxury furnished complex that students call expensive.",
        "metadata": {"doc_id": "05_collegemagazine", "source": "documents/05_collegemagazine.html"},
        "cosine_similarity": 0.6,
    }]


def _patch(monkeypatch, results, answer="An answer.", rewritten="REWRITTEN"):
    """Patch retrieve, _generate, and condense_question; capture their inputs.

    Args:
        monkeypatch: pytest fixture.
        results (list): Results that retrieve() should return.
        answer (str): Text that _generate() should return.
        rewritten (str): Standalone query that condense_question() should return.

    Returns:
        dict: A capture dict populated with retrieve_query, gen_messages, and
              condense_history on use.
    """
    cap = {}

    def fake_retrieve(q, *a, **k):
        cap["retrieve_query"] = q
        return {"query": q, "results": results, "reranked": True, "hybrid": True}

    def fake_generate(messages, llm_model, max_tokens, temperature):
        cap["gen_messages"] = messages
        return answer

    def fake_condense(question, history, llm_model=query.LLM_MODEL, max_turns=6):
        cap["condense_history"] = history
        return rewritten if history else question

    monkeypatch.setattr(query, "retrieve", fake_retrieve)
    monkeypatch.setattr(query, "_generate", fake_generate)
    monkeypatch.setattr(query, "condense_question", fake_condense)
    return cap


# ---- Tests -------------------------------------------------------------------
def test_followup_is_condensed_before_retrieval(monkeypatch) -> None:
    """Verify a follow-up is condensed and the condensed query drives retrieval.

    On the second turn (history non-empty), retrieve() must receive the rewritten
    standalone query, not the raw "Is it expensive?".
    """
    cap = _patch(monkeypatch, _fake_results(), rewritten="Is The Maxxen expensive?")
    conv = Conversation()

    conv.ask("Tell me about The Maxxen")
    assert cap["retrieve_query"] == "Tell me about The Maxxen"  # first turn: no history

    conv.ask("Is it expensive?")
    assert cap["retrieve_query"] == "Is The Maxxen expensive?"  # condensed via history


def test_history_is_injected_into_generation_prompt(monkeypatch) -> None:
    """Verify prior turns appear in the generation system prompt on a follow-up."""
    cap = _patch(monkeypatch, _fake_results(), answer="Yes, it is expensive [1].")
    conv = Conversation()
    conv.ask("Tell me about The Maxxen")
    conv.ask("Is it expensive?")

    system = cap["gen_messages"][0]["content"]
    assert "Conversation so far:" in system
    assert "The Maxxen" in system          # the earlier question is present
    # Grounding language survives alongside the history block.
    assert "ONLY the numbered context passages" in system


def test_grounding_preserved_on_empty_retrieval(monkeypatch) -> None:
    """Verify that empty retrieval still refuses, even mid-conversation."""
    _patch(monkeypatch, [])  # retrieval returns nothing
    conv = Conversation()
    conv.ask("Tell me about The Maxxen")
    res = conv.ask("Is it expensive?")
    assert res["grounded"] is False
    assert res["sources"] == []


def test_history_accumulates_and_window_is_bounded(monkeypatch) -> None:
    """Verify turns accumulate and only max_history_turns are forwarded to ask()."""
    cap = _patch(monkeypatch, _fake_results())
    conv = Conversation(max_history_turns=2)
    for i in range(4):
        conv.ask(f"question {i}")

    # All four turns are retained in the object's history...
    assert len(conv.history) == 4
    # ...but the last call only forwarded the most recent 2 prior turns to condense.
    assert len(cap["condense_history"]) == 2


def test_reset_clears_history(monkeypatch) -> None:
    """Verify reset() empties the conversation history."""
    _patch(monkeypatch, _fake_results())
    conv = Conversation()
    conv.ask("Tell me about The Maxxen")
    assert conv.history
    conv.reset()
    assert conv.history == []
