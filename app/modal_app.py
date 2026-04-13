"""
Phase 5 — Modal App Entry Point
Serves the MechRabot pipeline on a T4 GPU as a web endpoint.

TODO: implement in Phase 5 after pipeline is tested locally
"""

import modal
from app.config import MODAL_APP_NAME, MODAL_GPU

# ── Container image — all dependencies installed here ──────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "FlagEmbedding",
        "qdrant-client",
        "haystack-ai",
        "torch",
        "accelerate",
        "google-generativeai",
    )
)

app = modal.App(MODAL_APP_NAME, image=image)


# ── GPU Service Class ───────────────────────────────────────────────────────
@app.cls(gpu=MODAL_GPU)
class MechRabotService:

    @modal.enter()
    def setup(self):
        # Phase 5: build_pipeline() goes here — runs ONCE per container
        raise NotImplementedError("Phase 5 — not implemented yet")

    @modal.method()
    def query(self, text: str) -> dict:
        # Phase 5: pipe.run() goes here
        raise NotImplementedError("Phase 5 — not implemented yet")


# ── Web Endpoint ────────────────────────────────────────────────────────────
@app.function()
@modal.web_endpoint(method="POST")
def query_endpoint(request: dict) -> dict:
    """
    POST {"query": "your question"}
    Returns {"answer": "...", "sources": [...]}
    """
    # Phase 5: MechRabotService().query.remote(request["query"])
    raise NotImplementedError("Phase 5 — not implemented yet")
