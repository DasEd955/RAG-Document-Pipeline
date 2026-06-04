"""
Stage 5 — Interface: minimal Gradio UI over the grounded RAG pipeline.

The UI is intentionally thin. All grounding and source-attribution logic lives in
query.ask(); this module only renders the answer and the programmatically-built
source list. The "Retrieved from" box is populated from result["sources"], which
query.py derives from chunk metadata — it is never parsed from the model output.
"""
import gradio as gr

from query import ask


def handle_query(question: str):
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


with gr.Blocks(title="Unofficial Off-Campus Housing Guide (RAG)") as demo:
    gr.Markdown(
        "# Unofficial Off-Campus Housing Guide\n"
        "Grounded answers about off-campus housing near Penn State / State College, PA. "
        "Every answer is generated **only** from retrieved source documents, with citations."
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What do students say about Hendricks Investments properties?",
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
