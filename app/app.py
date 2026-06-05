"""
App.py - Minimal Gradio chat UI over the grounded, multi-turn RAG pipeline.

The UI is intentionally thin. All grounding, source-attribution, and
conversational-memory logic lives in app.query.ask() and app.conversation; this
module only renders the dialogue and the programmatically-built source list. The
"Retrieved from" box is populated from result["sources"], which query.py derives
from chunk metadata; it is never parsed from the model output.

A per-session Conversation object (held in gr.State) gives the chat memory: a
follow-up like "Is it expensive?" is resolved against earlier turns, while every
answer remains grounded solely in the retrieved passages.
"""
import os
import sys

import gradio as gr

# Make the project root importable so `app.*` (and `pipeline`) resolve whether
# launched as `python app/app.py` or `python -m app.app` from the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.conversation import Conversation


def respond(message: str, chat_history: list, conv: "Conversation | None"):
    """Answer one chat turn in context and update the dialogue + sources panel.

    Lazily creates a per-session Conversation (so memory is scoped to one browser
    session), runs the turn through it, appends the user/assistant messages to the
    chat transcript, and renders the latest turn's programmatic sources.

    Args:
        message (str): The user's latest message.
        chat_history (list): The running list of {"role", "content"} message dicts
            backing the gr.Chatbot.
        conv (Conversation | None): The session's Conversation, or None on the first
            turn.

    Returns:
        tuple: (chat_history, sources_text, conv, "") where the trailing ""
            clears the input textbox. conv is returned so gr.State persists it.
    """
    if conv is None:
        conv = Conversation()
    if not message or not message.strip():
        return chat_history, "", conv, ""

    result = conv.ask(message)
    answer = result["answer"]
    sources = result["sources"]
    sources_text = "\n".join(f"• {s}" for s in sources) if sources else "(no sources cited)"

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, sources_text, conv, ""


def reset(conv: "Conversation | None"):
    """Clear the chat transcript and the conversation memory.

    Args:
        conv (Conversation | None): The session's Conversation, if any.

    Returns:
        tuple: (empty_history, cleared_sources_text, fresh_conversation).
    """
    if conv is not None:
        conv.reset()
    return [], "", Conversation()


# Gradio UI definition: a chat window with memory, plus a panel showing the
# programmatic source attribution for the most recent answer.
with gr.Blocks(title="Unofficial Off-Campus Housing Guide (RAG)") as demo:
    gr.Markdown(
        "# Unofficial Off-Campus Housing Guide\n"
        "Grounded, multi-turn answers about off-campus housing near Penn State University & "
        "State College, PA. Every answer is generated **only** from retrieved source documents, "
        "with citations. Ask follow-ups (\"is it expensive?\", \"what about parking there?\") and "
        "the assistant remembers the conversation."
    )

    conv_state = gr.State(value=None)

    chatbot = gr.Chatbot(label="Conversation", height=420)
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. Is The Maxxen well ranked by students?  (then: is it expensive?)",
        lines=2,
    )
    sources = gr.Textbox(label="Retrieved from (latest answer)", lines=6)
    with gr.Row():
        btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear conversation")

    btn.click(respond, inputs=[inp, chatbot, conv_state], outputs=[chatbot, sources, conv_state, inp])
    inp.submit(respond, inputs=[inp, chatbot, conv_state], outputs=[chatbot, sources, conv_state, inp])
    clear_btn.click(reset, inputs=[conv_state], outputs=[chatbot, sources, conv_state])

    gr.Examples(
        examples=[
            "What do students say about Hendricks Investments properties?",
            "Is downtown State College, PA expensive?",
            "Do you have to act fast to get off-campus housing at Penn State?",
            "Is The Maxxen well ranked by students?",
            "Can I feasibly live off-campus with no car?",
        ],
        inputs=inp,
    )


if __name__ == "__main__":
    demo.launch()
