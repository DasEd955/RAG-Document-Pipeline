"""
App.py - Minimal Gradio UI Interfaceover the grounded RAG pipeline.

The UI is intentionally thin. All grounding and source-attribution logic lives in
query.ask(); this module only renders the answer and the programmatically-built
source list. The "Retrieved from" box is populated from result["sources"], which
query.py derives from chunk metadata; it is never parsed from the model output.
"""
import os
import sys
import gradio as gr
from app.query import ask

# Make the project root importable so `app.query` (and `pipeline`) resolve whether
# launched as `python app/app.py` or `python -m app.app` from the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

def handle_query(question: str) -> tuple:
    """Process a user query and return grounded answer + sources for Gradio.

    Calls ask() to retrieve context and generate a grounded answer, then formats
    the result for display. Returns a tuple of (answer_text, sources_text) for the
    Gradio output textboxes.

    Args:
        question (str): The user's question string.

    Returns:
        tuple: A 2-tuple of (answer, sources_text) for Gradio output boxes.
               - answer (str): The grounded answer or refusal phrase.
               - sources_text (str): Newline-separated source attribution list,
                                     or "(no sources cited)" if no grounding.

    Example:
        >>> answer, sources = handle_query("Is downtown expensive?")
        >>> print(answer[:50])
        Yes, downtown State College is expensive...
    """
    if not question or not question.strip():
        return "Please enter a question.", ""
    result = ask(question)
    answer = result["answer"]
    sources = result["sources"]
    if sources:
        sources_text = "\n".join(f"• {s}" for s in sources)
    else:
        # Refusal or empty retrieval: no grounded claim, so no attribution.
        sources_text = "(no sources cited)"
    return answer, sources_text
 

# Gradio UI definition: a simple interface with a question box, an "Ask" button, and output boxes for the answer and sources.
with gr.Blocks(title="Unofficial Off-Campus Housing Guide (RAG)") as demo:
    gr.Markdown(
        "# Unofficial Off-Campus Housing Guide\n"
        "Grounded answers about off-campus housing near Penn State University & State College, PA. "
        "Every answer is generated **only** from retrieved source documents, with citations."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. Is downtown State College, PA expensive?",
        lines=2,
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=6)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

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
