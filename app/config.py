"""
MechRabot — Central Configuration
All constants live here. Secrets are loaded from environment variables
so they work both locally (.env) and on Modal (Modal Secrets).
"""

import os

# ── Qdrant ─────────────────────────────────────────────────────────────────
_qdrant_url       = os.environ.get("QDRANT_URL", "")
_qdrant_port      = os.environ.get("QDRANT_PORT", "")
QDRANT_URL        = f"{_qdrant_url}:{_qdrant_port}" if _qdrant_port else _qdrant_url
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "mechrabot_Vdb_1"

# ── Embedding Model ────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "BAAI/bge-m3"
EMBEDDING_USE_FP16 = True          # set False if running on CPU
EMBEDDING_BATCH_SIZE = 1           # 1 at inference time (single query)
EMBEDDING_MAX_LENGTH = 512

# ── Retrieval ──────────────────────────────────────────────────────────────
PREFETCH_LIMIT    = 30             # candidates from sparse + dense each
FINAL_TOP_K       = 5              # final results after ColBERT elevation

# ── LLM (Generator) ────────────────────────────────────────────────────────
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL      = "gemini-2.0-flash"

# ── Modal ──────────────────────────────────────────────────────────────────
MODAL_APP_NAME    = "mechrabot"
MODAL_GPU         = "T4"

# ── Image Storage (HuggingFace Dataset) ────────────────────────────────────
# 1285 extracted manual images hosted on HF
# payload field: linked_images = ["extracted_artifacts/image_XXXXXX_<hash>.png"]
# → strip prefix → build URL below
HF_IMAGES_BASE_URL = "https://huggingface.co/datasets/ahmedezzattaha/mechrabot-images/resolve/main"
