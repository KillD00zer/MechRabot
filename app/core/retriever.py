"""
MechRabotRetriever — Haystack component
"""

from haystack import component, Document
from typing import List
from qdrant_client import QdrantClient, models


def prepare_sparse_vector(single_sparse_dict):
    """
    Convert a single sparse vector dictionary into a Qdrant SparseVector object.
    """
    indices = [int(token_id) for token_id in single_sparse_dict.keys()]
    values = list(single_sparse_dict.values())
    return models.SparseVector(indices=indices, values=values)


@component
class MechRabotRetriever:
    """Hybrid retriever: BGE-M3 (dense + sparse + colbert) over Qdrant."""

    def __init__(self, client, col_name, prefetch_limit=50, top_k=15):
        self.client = client
        self.col_name = col_name
        self.prefetch_limit = prefetch_limit
        self.top_k = top_k

    @component.output_types(documents=List[Document])  #list of haystack documents 1 for each query
    def run(self, sparse_dict: dict, dense_list: list, colbert_list: list):

        searchers = [models.Prefetch(query=prepare_sparse_vector(sparse_dict), using="sparse", limit=self.prefetch_limit),
                     models.Prefetch(query=dense_list, using="dense", limit=self.prefetch_limit)
                    ]     # <--------- define where to search

        prefetcher = models.Prefetch(prefetch=searchers,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=self.prefetch_limit)                 # <-------- the define reranker RRF to rerank top 50 retrievals

        response = self.client.query_points(                       # <--------- starting the process
            collection_name=self.col_name,
            prefetch=prefetcher,
            query=colbert_list,                    # <----  use colbert to elevat top_k of the 50
            using="colbert",
            limit=self.top_k)

        # convert Qdrant points → Haystack Documents
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
