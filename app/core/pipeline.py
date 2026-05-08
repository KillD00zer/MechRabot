"""
MechRabot Pipeline
Wires all components into a single Haystack Pipeline.

Flow:
  query → refiner_prompt → refiner_llm → embedder → retriever → generator_prompt → generator_llm → answer
"""

from haystack import Pipeline
from qdrant_client import QdrantClient
import os

from app.core.prompt_refiner import translator_refiner_template
from app.core.prompt_generator import generator_template
from app.core.embedder import MechRabotEmbedder
from app.core.retriever import MechRabotRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator


def build_pipeline() -> Pipeline:
    """
    Factory function — returns a fully wired MechRabot pipeline.

    Usage:
        pipe = build_pipeline()
        result = pipe.run({"refiner_prompt": {"query": "your question here"}})
        print(result["generator_llm"]["replies"][0].text)
    """
    pipe = Pipeline()

    # ── 1. Refiner: translate + refine the query ──────────────────────
    pipe.add_component("refiner_prompt", ChatPromptBuilder(template=translator_refiner_template))
    pipe.add_component("refiner_llm",   GoogleAIGeminiChatGenerator(model="gemini-1.5-flash", generation_config={"temperature": 0.2}))

    # ── 2. Embedder: BGE-M3 → sparse + dense + colbert ───────────────
    pipe.add_component("embedder", MechRabotEmbedder())

    # ── 3. Retriever: Qdrant hybrid search ────────────────────────────
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    pipe.add_component("retriever", MechRabotRetriever(
        client=client, col_name="mechrabot_Vdb_1",
    ))

    # ── 4. Generator: final answer ────────────────────────────────────
    pipe.add_component("generator_prompt", ChatPromptBuilder(template=generator_template))
    pipe.add_component("generator_llm",   GoogleAIGeminiChatGenerator(model="gemini-2.5-pro-preview-05-06", generation_config={"temperature": 0.1}))

    # ── Connections ───────────────────────────────────────────────────
    pipe.connect("refiner_prompt.prompt", "refiner_llm.messages")
    pipe.connect("refiner_llm.replies",   "embedder.query")
    pipe.connect("embedder.sparse_dict",  "retriever.sparse_dict")
    pipe.connect("embedder.dense_list",   "retriever.dense_list")
    pipe.connect("embedder.colbert_list", "retriever.colbert_list")
    pipe.connect("retriever.documents",   "generator_prompt.documents")
    pipe.connect("refiner_llm.replies",   "generator_prompt.query")
    pipe.connect("generator_prompt.prompt", "generator_llm.messages")

    return pipe
