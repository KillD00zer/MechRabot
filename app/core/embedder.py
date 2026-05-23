"""
MechRabotEmbedder — Haystack component
"""

from haystack import component
from typing import Dict, List, Any
from FlagEmbedding import BGEM3FlagModel


@component
class MechRabotEmbedder:
    """BGE-M3 embedder: query → sparse + dense + colbert vectors."""

    def __init__(self, model=None, model_name='BAAI/bge-m3', use_fp16=True, batch_size=64, max_length=512):
        if model is not None:
            self.model = model
        else:
            self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.batch_size = batch_size
        self.max_length = max_length

    @component.output_types(sparse_dict=dict, dense_list=list, colbert_list=list)
    def run(self, query: Any):
        
        # Extract text if query is a list of ChatMessage objects
        if isinstance(query, list):
            query = query[0].text

        # Always pass as a list so the model returns consistent dictionaries
        embedding_docu = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        return {
            "sparse_dict":  embedding_docu["lexical_weights"][0],
            "dense_list":   embedding_docu["dense_vecs"][0].tolist(),
            "colbert_list": embedding_docu["colbert_vecs"][0].tolist(),
        }
