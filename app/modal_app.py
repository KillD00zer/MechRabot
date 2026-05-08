"""
MechRabot — Modal App
Serves the Haystack pipeline on a T4 GPU as a web endpoint.
"""

import modal

# ── Container image ─────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "FlagEmbedding",
        "qdrant-client",
        "haystack-ai",
        "google-genai-haystack",
        "torch",
        "accelerate",
    )
)

app = modal.App("mechrabot", image=image)


# ── GPU Service ─────────────────────────────────────────────────────────────
@app.cls(
    gpu="T4",
    secrets=[
        modal.Secret.from_name("google-gen-api-mechrabot"),   # GEMINI_API_KEY
        modal.Secret.from_name("qdrant-secret-mechrabot"),    # QDRANT_URL, QDRANT_PORT, QDRANT_API_KEY
    ],
)
class MechRabotService:

    @modal.enter()
    def setup(self):
        from app.core.pipeline import build_pipeline
        self.pipe = build_pipeline()

    @modal.method()
    def query(self, text: str) -> dict:
        result = self.pipe.run({
            "refiner_prompt":   {"query": text},
            "generator_prompt": {"query": text},
        })

        answer = result["generator_llm"]["replies"][0].text
        documents = result["retriever"]["documents"]

        sources = []
        for doc in documents:
            sources.append({
                "chunk_id": doc.id,
                "score": doc.score,
                "page_no": doc.meta.get("page_no"),
                "source_file": doc.meta.get("source_file"),
                "section_path": doc.meta.get("section_path"),
                "linked_images": doc.meta.get("linked_images", []),
            })

        return {"answer": answer, "sources": sources}


# ── Web Endpoint ────────────────────────────────────────────────────────────
@app.function(
    secrets=[
        modal.Secret.from_name("google-gen-api-mechrabot"),
        modal.Secret.from_name("qdrant-secret-mechrabot"),
    ],
)
@modal.fastapi_endpoint(method="POST")
def query_endpoint(request: dict) -> dict:
    """
    POST {"query": "your question"}
    Returns {"answer": "...", "sources": [...]}
    """
    service = MechRabotService()
    return service.query.remote(request["query"])


# ── Test Entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def test(text: str = "What is the torque for cylinder head cover bolts?"):
    service = MechRabotService()
    result = service.query.remote(text)
    print(result["answer"])
