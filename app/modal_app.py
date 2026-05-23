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
cache_volume  = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
images_volume = modal.Volume.from_name("mechrabot-images")
IMAGES_DIR = "/images"


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
        import os
        from qdrant_client import QdrantClient
        from app.core.pipeline import build_pipeline

        # 1. Instantiate the heavy embedding model weights ONCE in GPU VRAM
        print("Loading BGE-M3 model weights into GPU VRAM...")
        bge_model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

        # 2. Instantiate a shared Qdrant client
        print("Initializing Qdrant client...")
        qdrant_client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

        # 3. Build pipelines sharing the loaded model and client
        self.pipe_standard = build_pipeline(use_reasoner=False, model=bge_model, client=qdrant_client)
        self.pipe_super = build_pipeline(use_reasoner=True, model=bge_model, client=qdrant_client)

    @modal.method()
    def query(self, text: str, mode: str = "restricted") -> dict:
        pipe = self.pipe_super if mode == "super_augmented" else self.pipe_standard
        prompt_mode = "augmented" if mode == "super_augmented" else mode

        result = pipe.run({
            "refiner_prompt":   {"query": text},
            "generator_prompt": {"query": text, "mode": prompt_mode},
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
    volumes={IMAGES_DIR: images_volume},
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


# ── Image Endpoint ──────────────────────────────────────────────────────────
@app.function(volumes={IMAGES_DIR: images_volume})
@modal.fastapi_endpoint(method="GET")
def get_image(name: str):
    """
    GET ?name=image_000361
    Returns the PNG file from the mechrabot-images volume.
    """
    from fastapi.responses import FileResponse, JSONResponse
    from pathlib import Path

    path = Path(IMAGES_DIR) / f"{name}.png"
    if not path.exists():
        return JSONResponse({"error": f"Image '{name}' not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


# ── Test Entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def test(text: str = " مقاس الجنط؟"):
    service = MechRabotService()
    result = service.query.remote(text)
    print(result["answer"])
