"""
Phase 4 — MechRabot Pipeline
Assembles retriever + generator into a single Haystack Pipeline.
Call build_pipeline() to get a ready-to-run pipeline instance.

TODO: uncomment imports after Phase 2 and Phase 3 are done
"""

from haystack import Pipeline

# from app.core.retriever import MechRabotRetriever   # uncomment in Phase 4
# from app.core.generator import MechRabotGenerator   # uncomment in Phase 4


def build_pipeline() -> Pipeline:
    """
    Factory function — returns a fully wired MechRabot pipeline.
    Usage:
        pipe = build_pipeline()
        result = pipe.run({"retriever": {"query": "your question here"}})
        print(result["generator"]["answer"])
    """
    # Phase 4: wire components
    raise NotImplementedError("Phase 4 — not implemented yet")
