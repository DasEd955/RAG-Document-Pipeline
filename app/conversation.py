"""Conversation.py - Stateful multi-turn wrapper around the grounded ask() pipeline.

Holds a rolling conversation history and feeds it into app.query.ask() on each
turn so that follow-up questions ("Is it expensive?", "What about parking there?")
are understood in context. The two memory effects are owned by query.py:
  1. Retrieval: the follow-up is condensed into a standalone query before search.
  2. Generation: prior turns are added to the prompt for reference resolution.

The grounding contract is unchanged: every answer is still produced solely from
the retrieved passages, with programmatic source attribution. This module only
adds state (the turn history); it does not relax grounding.

Run an interactive session: python -m app.conversation
"""
import os
import sys
from typing import Any, Dict, List, Optional

# Make the project root importable so `app.query` (and `pipeline`) resolve whether
# launched as `python app/conversation.py` or `python -m app.conversation`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.query import ask, DEFAULT_K


class Conversation:
    """A multi-turn session that remembers prior turns across calls to ask().

    Each call appends a {"question", "answer"} turn to an internal history list and
    forwards a bounded window of that history to app.query.ask(), which handles
    follow-up condensation and history-aware generation. Construction-time keyword
    arguments (k, filters, source_boost, hybrid, etc.) are stored and applied to
    every turn.

    Attributes:
        history (List[Dict[str, str]]): Accumulated turns, oldest first.
        max_history_turns (int): Upper bound on turns forwarded to ask() per call.
        ask_kwargs (Dict[str, Any]): Per-turn keyword arguments passed to ask().

    Example:
        >>> conv = Conversation(k=5)  # doctest: +SKIP
        >>> conv.ask("Tell me about The Maxxen")["answer"]      # doctest: +SKIP
        >>> conv.ask("Is it expensive?")["answer"]              # doctest: +SKIP
    """

    def __init__(self, *, max_history_turns: int = 6, **ask_kwargs: Any) -> None:
        """Initialize an empty conversation.

        Args:
            max_history_turns (int, optional): Maximum number of trailing turns to
                forward to ask() on each call. Defaults to 6.
            **ask_kwargs (Any): Keyword arguments forwarded to ask() on every turn
                (e.g., k, persist_dir, collection_name, hybrid, filters, source_boost).
        """
        self.history: List[Dict[str, str]] = []
        self.max_history_turns = max_history_turns
        self.ask_kwargs = ask_kwargs

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question in the context of the conversation so far, then record it.

        Forwards the bounded history window to ask(), appends the resulting turn to
        the history, and returns the full result dict unchanged.

        Args:
            question (str): The user's latest message.

        Returns:
            Dict[str, Any]: The result dict from ask() (answer, sources, grounded,
                chunks, retrieval_query).
        """
        window = self.history[-self.max_history_turns:]
        result = ask(question, history=window, **self.ask_kwargs)
        self.history.append({"question": question, "answer": result.get("answer", "")})
        return result

    def reset(self) -> None:
        """Clear all conversation history, starting a fresh session.

        Returns:
            None
        """
        self.history.clear()


# ---- Interactive REPL --------------------------------------------------------
def _cli() -> None:
    """Run an interactive multi-turn question-answering loop in the terminal.

    Reads questions from stdin until EOF or one of 'exit'/'quit'/'q'. Each turn
    prints the grounded answer and its programmatic sources, carrying conversation
    context across turns. 'reset' clears the history.
    """
    import argparse
    p = argparse.ArgumentParser(description="Interactive multi-turn grounded Q&A over the housing corpus.")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--persist_dir", default="chroma_db")
    p.add_argument("--collection", default="documents")
    p.add_argument("--no-hybrid", action="store_true", help="Disable BM25 keyword recall (semantic only)")
    p.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    args = p.parse_args()

    conv = Conversation(
        k=args.k,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        hybrid=not args.no_hybrid,
        rerank=not args.no_rerank,
    )

    print("Multi-turn housing assistant. Type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        if question.lower() == "reset":
            conv.reset()
            print("(history cleared)\n")
            continue

        result = conv.ask(question)
        print(f"\nAssistant: {result['answer']}\n")
        if result.get("sources"):
            print("Retrieved from:")
            for s in result["sources"]:
                print(f"  {s}")
        print()


if __name__ == "__main__":
    _cli()
