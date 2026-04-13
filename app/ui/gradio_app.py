"""
Phase 6 — Gradio UI
Deployed to HuggingFace Spaces. Calls the Modal web endpoint.
No ML code here — this file is pure UI.

TODO: implement in Phase 6 after Modal endpoint is live
"""

import gradio as gr
import requests
import os

MODAL_ENDPOINT = os.environ.get("MODAL_ENDPOINT_URL", "")  # set in HF Space secrets


def ask_mechrabot(query: str):
    # Phase 6: calls Modal endpoint and formats the response
    raise NotImplementedError("Phase 6 — not implemented yet")


# ── UI Layout ───────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("# 🔧 MechRabot — Mechanical Manual Assistant")
    # Phase 6: build full UI here

if __name__ == "__main__":
    demo.launch()
