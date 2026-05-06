"""
MechRabot Pipeline
Wires all components into a single Haystack Pipeline.

Flow:
  query → refiner_prompt → refiner_llm → embedder → retriever → generator_prompt → generator_llm → answer
"""

from haystack import Pipeline
from qdrant_client import QdrantClient

from app.core.prompt_refiner import refiner_prompt_builder, gemini_refiner_agent
from app.core.prompt_generator import generator_prompt_builder, gemini_generator_agent
from app.core.embedder import MechRabotEmbedder
from app.core.retriever import MechRabotRetriever
from app.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION


def build_pipeline() -> Pipeline:
    """
    Factory function — returns a fully wired MechRabot pipeline.

    Usage:
        pipe = build_pipeline()
        result = pipe.run({"refiner_prompt": {"query": "your question here"},
                           "generator_prompt": {"query": "your question here"}})
        print(result["generator_llm"]["replies"][0].text)
    """
    pipe = Pipeline()

    # ── 1. Refiner: translate + refine the query ──────────────────────
    pipe.add_component("refiner_prompt", refiner_prompt_builder)
    pipe.add_component("refiner_llm", gemini_refiner_agent)

    # ── 2. Embedder: BGE-M3 → sparse + dense + colbert ───────────────
    pipe.add_component("embedder", MechRabotEmbedder())

    # ── 3. Retriever: Qdrant hybrid search ────────────────────────────
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    pipe.add_component("retriever", MechRabotRetriever(
        client=client, col_name=QDRANT_COLLECTION,
    ))

    # ── 4. Generator: final answer ────────────────────────────────────
    pipe.add_component("generator_prompt", generator_prompt_builder)
    pipe.add_component("generator_llm", gemini_generator_agent)

    # ── Connections ───────────────────────────────────────────────────
    pipe.connect("refiner_prompt.prompt", "refiner_llm.messages")
    pipe.connect("refiner_llm.replies",   "embedder.query")
    pipe.connect("embedder.sparse_dict",  "retriever.sparse_dict")
    pipe.connect("embedder.dense_list",   "retriever.dense_list")
    pipe.connect("embedder.colbert_list", "retriever.colbert_list")
    pipe.connect("retriever.documents",   "generator_prompt.documents")
    pipe.connect("generator_prompt.prompt", "generator_llm.messages")

    return pipe
