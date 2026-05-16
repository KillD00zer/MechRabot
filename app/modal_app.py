"""
MechRabot — Modal App
Serves the Haystack pipeline on a T4 GPU as a web endpoint.
"""

import modal

# ── Container image ─────────────────────────────────────────────────────────
CACHE_DIR = "/hf-cache"
MODEL_NAME = "BAAI/bge-m3"
MINUTES = 60  # seconds

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "FlagEmbedding",
        "qdrant-client",
        "haystack-ai",
        "torch",
        "accelerate",
        "fastapi[standard]",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": CACHE_DIR})
    .add_local_python_source("app")
)

# Import heavy deps only inside the remote container
with image.imports():
    from FlagEmbedding import BGEM3FlagModel

app = modal.App("mechrabot", image=image)

# ── Volumes ─────────────────────────────────────────────────────────────────
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


# ── Cache model weights ────────────────────────────────────────────────────
@app.function(
    image=image, volumes={CACHE_DIR: cache_volume}, timeout=20 * MINUTES
)
def download_model():
    from huggingface_hub import snapshot_download

    result = snapshot_download(MODEL_NAME)
    print(f"Downloaded model weights to {result}")


# ── GPU Service ─────────────────────────────────────────────────────────────
@app.cls(
    gpu="T4",
    scaledown_window=10 * MINUTES,
    volumes={CACHE_DIR: cache_volume},
    secrets=[
        modal.Secret.from_name("qdrant-secret-mechrabot"),
        modal.Secret.from_name("deepseek_APi")
    ],
)
class MechRabotService:

    @modal.enter()
    def setup(self):
        from app.core.pipeline import build_pipeline
        self.pipe = build_pipeline()

    @modal.method()
    def query(self, text: str, mode: str = "restricted") -> dict:
        result = self.pipe.run({
            "refiner_prompt":   {"query": text},
            "generator_prompt": {"query": text, "mode": mode},
        }, include_outputs_from={"retriever"})

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
        modal.Secret.from_name("qdrant-secret-mechrabot"),
        modal.Secret.from_name("deepseek_APi")
    ],
)
@modal.fastapi_endpoint(method="POST")
def query_endpoint(request: dict) -> dict:
    """
    POST {"query": "your question", "mode": "restricted|augmented"}
    Returns {"answer": "...", "sources": [...]}
    """
    mode = request.get("mode", "restricted")
    service = MechRabotService()
    return service.query.remote(request["query"], mode)


# ── Test Entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def test(text: str = " مقاس الجنط؟"):
    service = MechRabotService()
    result = service.query.remote(text)
    print(result["answer"])
