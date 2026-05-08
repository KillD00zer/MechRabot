"""
MechRabot — Full Pipeline Script
=================================
Single-file replacement for the entire app/core/ package.

Replaces:
    app/core/prompt_refiner.py
    app/core/prompt_generator.py
    app/core/embedder.py
    app/core/retriever.py
    app/core/pipeline.py

Usage (Colab):
    from app.fullscript import build_pipeline

    pipe = build_pipeline()
    result = pipe.run({
        "refiner_prompt":   {"query": "your question"},
        "generator_prompt": {"query": "your question"},
    })
    print(result["generator_llm"]["replies"][0].text)
"""

import os
from typing import List

# ── Haystack ──────────────────────────────────────────────────────────────────
from haystack import Pipeline, component, Document
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# ── External ──────────────────────────────────────────────────────────────────
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models


# ══════════════════════════════════════════════════════════════════════════════
# 0.  CREDENTIALS  ← put your keys here
# ══════════════════════════════════════════════════════════════════════════════

os.environ["GEMINI_API_KEY"]  = "YOUR_GEMINI_API_KEY"
os.environ["QDRANT_URL"]      = "YOUR_QDRANT_URL"        # e.g. https://xxxx.qdrant.io:6333
os.environ["QDRANT_API_KEY"]  = "YOUR_QDRANT_API_KEY"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

translator_refiner_template = [ChatMessage.from_user(
    """\
You are a search query optimizer for an automotive repair manual (chery m11).

If the query is not in English, translate it to English first.
Then rewrite it into a clean, precise search query.
- Remove filler words
- Keep all technical terms (torque, N·m, bolt, DTC, etc.)
- Be concise

Query: {{ query }}

Return only the refined English query, nothing else.\
"""
)]

generator_template = [ChatMessage.from_user(
    """\
You are MechRabot 🤖, an expert Chery M11 automotive repair assistant.

You will receive a user's question and 5 retrieved chunks from the service manual.
Each chunk has:
- content: the actual text from the manual (specs, procedures, tables, diagnostics)
- metadata: source_file, page_no, chunk_type, section_path, linked_images

Your job:
1. Read all 5 chunks carefully
2. Extract the answer ONLY from the chunk content — do not make up information
3. If multiple chunks contribute to the answer, combine them
4. If none of the chunks contain the answer, say so honestly

Rules:
- Answer in the SAME language as the user's query
- If the query is in Egyptian slang, reply in Egyptian slang
- Include exact specs when available (torque values in N·m, DTC codes, measurements)
- Be clear and practical — you're talking to a mechanic
- At the end of your answer, add a "📎 Sources" section listing which chunks you used

User query: {{ query }}

Retrieved chunks:
{% for doc in documents %}
━━━━━━━━━━━━━━━━━━━━━━━━
Chunk {{ loop.index }} (score: {{ doc.score }})
Section: {{ doc.meta.section_path | join(" > ") }}
Source: {{ doc.meta.source_file }} — Page {{ doc.meta.page_no }}
Type: {{ doc.meta.chunk_type }}
{% if doc.meta.linked_images %}Images: {{ doc.meta.linked_images | join(", ") }}{% endif %}

Content:
{{ doc.content }}
━━━━━━━━━━━━━━━━━━━━━━━━
{% endfor %}

Answer:\
"""
)]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EMBEDDER COMPONENT
# ══════════════════════════════════════════════════════════════════════════════

@component
class MechRabotEmbedder:
    """BGE-M3 embedder: query string → sparse + dense + colbert vectors."""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True,
                 batch_size: int = 64, max_length: int = 512):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length

    @component.output_types(sparse_dict=dict, dense_list=list, colbert_list=list)
    def run(self, query: str):
        embedding = self.model.encode(
            query,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        return {
            "sparse_dict":  embedding["lexical_weights"][0],
            "dense_list":   embedding["dense_vecs"][0].tolist(),
            "colbert_list": embedding["colbert_vecs"][0].tolist(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RETRIEVER COMPONENT
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_sparse_vector(sparse_dict: dict) -> models.SparseVector:
    """Convert sparse-weights dict → Qdrant SparseVector."""
    indices = [int(k) for k in sparse_dict.keys()]
    values  = list(sparse_dict.values())
    return models.SparseVector(indices=indices, values=values)


@component
class MechRabotRetriever:
    """Hybrid retriever: BGE-M3 (sparse + dense + colbert re-rank) over Qdrant."""

    def __init__(self, client: QdrantClient, col_name: str,
                 prefetch_limit: int = 50, top_k: int = 5):
        self.client         = client
        self.col_name       = col_name
        self.prefetch_limit = prefetch_limit
        self.top_k          = top_k

    @component.output_types(documents=List[Document])
    def run(self, sparse_dict: dict, dense_list: list, colbert_list: list):
        # Stage 1 — sparse + dense prefetch
        searchers = [
            models.Prefetch(query=_prepare_sparse_vector(sparse_dict),
                            using="sparse", limit=self.prefetch_limit),
            models.Prefetch(query=dense_list,
                            using="dense",  limit=self.prefetch_limit),
        ]

        # Stage 2 — RRF fusion
        prefetcher = models.Prefetch(
            prefetch=searchers,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.prefetch_limit,
        )

        # Stage 3 — ColBERT re-rank → top_k
        response = self.client.query_points(
            collection_name=self.col_name,
            prefetch=prefetcher,
            query=colbert_list,
            using="colbert",
            limit=self.top_k,
        )

        # Convert Qdrant points → Haystack Documents
        documents = []
        for point in response.points:
            payload = point.payload or {}
            documents.append(Document(
                id=str(point.id),
                content=payload.get("content", ""),
                score=point.score,
                meta={k: v for k, v in payload.items() if k != "content"},
            ))

        return {"documents": documents}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PIPELINE FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "mechrabot_Vdb_1",
) -> Pipeline:
    """
    Build and return a fully wired MechRabot pipeline.

    All component instances are created fresh on every call (Haystack requirement).

    Args:
        qdrant_url:      Qdrant cluster URL. Falls back to QDRANT_URL env var.
        qdrant_api_key:  Qdrant API key.  Falls back to QDRANT_API_KEY env var.
        collection_name: Qdrant collection name. Default: 'mechrabot_Vdb_1'.

    Returns:
        A ready-to-run Haystack Pipeline.

    Example:
        pipe = build_pipeline()
        result = pipe.run({
            "refiner_prompt":   {"query": "engine oil specs"},
            "generator_prompt": {"query": "engine oil specs"},
        })
        print(result["generator_llm"]["replies"][0].text)
    """
    url     = qdrant_url     or os.environ["QDRANT_URL"]
    api_key = qdrant_api_key or os.environ["QDRANT_API_KEY"]

    pipe = Pipeline()

    # ── 1. Refiner ────────────────────────────────────────────────────
    pipe.add_component("refiner_prompt",
                       ChatPromptBuilder(template=translator_refiner_template))
    pipe.add_component("refiner_llm",
                       GoogleGenAIChatGenerator(model="gemini-2.0-flash"))

    # ── 2. Embedder ───────────────────────────────────────────────────
    pipe.add_component("embedder", MechRabotEmbedder())

    # ── 3. Retriever ──────────────────────────────────────────────────
    client = QdrantClient(url=url, api_key=api_key)
    pipe.add_component("retriever", MechRabotRetriever(
        client=client, col_name=collection_name,
    ))

    # ── 4. Generator ──────────────────────────────────────────────────
    pipe.add_component("generator_prompt",
                       ChatPromptBuilder(template=generator_template))
    pipe.add_component("generator_llm",
                       GoogleGenAIChatGenerator(model="gemini-2.5-pro-preview-05-06"))

    # ── Connections ───────────────────────────────────────────────────
    pipe.connect("refiner_prompt.prompt",   "refiner_llm.messages")
    pipe.connect("refiner_llm.replies",     "embedder.query")
    pipe.connect("embedder.sparse_dict",    "retriever.sparse_dict")
    pipe.connect("embedder.dense_list",     "retriever.dense_list")
    pipe.connect("embedder.colbert_list",   "retriever.colbert_list")
    pipe.connect("retriever.documents",     "generator_prompt.documents")
    pipe.connect("generator_prompt.prompt", "generator_llm.messages")

    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# 5.  INTERACTIVE TEST FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def ask(query: str, pipe: Pipeline | None = None) -> str:
    """
    Run a single query through the pipeline and return the answer text.

    Args:
        query: Your question in any language.
        pipe:  An already-built pipeline. If None, build_pipeline() is called.

    Returns:
        The answer string from the generator LLM.
    """
    if pipe is None:
        pipe = build_pipeline()
    result = pipe.run({
        "refiner_prompt":   {"query": query},
        "generator_prompt": {"query": query},
    })
    return result["generator_llm"]["replies"][0].text


def chat_loop():
    """
    Interactive REPL — type your question, get an answer.
    Type 'exit' or 'quit' to stop.

    Usage:
        from app.fullscript import chat_loop
        chat_loop()
    """
    print("🤖 MechRabot — Chery M11 Assistant")
    print("   Type your question (English / Arabic / Egyptian slang)")
    print("   Type 'exit' to quit\n")

    pipe = build_pipeline()          # build once, reuse for all queries

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "bye"}:
            print("👋 Bye!")
            break

        print("\n🔄 Thinking...\n")
        try:
            answer = ask(query, pipe=pipe)
            print(f"MechRabot:\n{answer}\n")
            print("─" * 60 + "\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    chat_loop()
