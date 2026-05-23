"""
MechRabot Pipeline
Wires all components into a single Haystack Pipeline.

Flow:
  query → refiner_prompt → refiner_llm → embedder → retriever → generator_prompt → generator_llm → answer
"""

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from qdrant_client import QdrantClient
import os

from app.core.prompt_refiner import translator_refiner_template
from app.core.prompt_generator import generator_template
from app.core.embedder import MechRabotEmbedder
from app.core.retriever import MechRabotRetriever


def build_pipeline(use_reasoner: bool = False, model = None, client = None) -> Pipeline:
    """
    Factory function — returns a fully wired MechRabot pipeline.
    Each call creates fresh component instances (required by Haystack).

    Usage:
        pipe = build_pipeline(use_reasoner=True)
        result = pipe.run({"refiner_prompt": {"query": "your question here"},
                           "generator_prompt": {"query": "your question here", "mode": "augmented"}})
        print(result["generator_llm"]["replies"][0].text)
    """
    pipe = Pipeline()

    # ── 1. Refiner: translate + refine the query ──────────────────────
    pipe.add_component("refiner_prompt", ChatPromptBuilder(template=translator_refiner_template, required_variables=["query"]))
    pipe.add_component("refiner_llm", OpenAIChatGenerator(
        model="deepseek-v4-flash",
        api_key=Secret.from_env_var("deepseek_APi"),
        api_base_url="https://api.deepseek.com",
        generation_kwargs={},
    ))

    # ── 2. Embedder: BGE-M3 → sparse + dense + colbert ───────────────
    pipe.add_component("embedder", MechRabotEmbedder(model=model))

    # ── 3. Retriever: Qdrant hybrid search ────────────────────────────
    if client is None:
        client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    pipe.add_component("retriever", MechRabotRetriever(
        client=client, col_name="mechrabot_Vdb_1",
    ))

    # ── 4. Generator: final answer ────────────────────────────────────
    gen_model = "deepseek-v4-pro" if use_reasoner else "deepseek-v4-flash"
    gen_kwargs = {} if use_reasoner else {"temperature": 0.4}
    pipe.add_component("generator_prompt", ChatPromptBuilder(template=generator_template, required_variables=["documents", "query", "mode"]))
    pipe.add_component("generator_llm", OpenAIChatGenerator(
        model=gen_model,
        api_key=Secret.from_env_var("deepseek_APi"),
        api_base_url="https://api.deepseek.com",
        generation_kwargs=gen_kwargs,
    ))

    # ── Connections ───────────────────────────────────────────────────
    pipe.connect("refiner_prompt.prompt", "refiner_llm.messages")
    pipe.connect("refiner_llm.replies",   "embedder.query")
    pipe.connect("embedder.sparse_dict",  "retriever.sparse_dict")
    pipe.connect("embedder.dense_list",   "retriever.dense_list")
    pipe.connect("embedder.colbert_list", "retriever.colbert_list")
    pipe.connect("retriever.documents",   "generator_prompt.documents")
    pipe.connect("generator_prompt.prompt", "generator_llm.messages")

    return pipe
