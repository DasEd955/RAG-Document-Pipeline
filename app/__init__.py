"""
Application Layer: Grounded generation and Gradio UI; __init__.py for app package.

Exports ask() for end-to-end generation, and demo for Gradio interface.
Both enforce strict grounding: answers come only from retrieved context,
with sources built programmatically from chunk metadata.
"""
