"""
Phase 3 — MechRabotGenerator
Custom Haystack component that takes retrieved Documents and generates
a final answer using the Gemini API.

TODO: implement in Phase 3
"""

from haystack import component, Document
from typing import List


@component
class MechRabotGenerator:
    """Answer generator: retrieved chunks → LLM → final answer string."""

    def __init__(self):
        # Phase 3: init Gemini client
        raise NotImplementedError("Phase 3 — not implemented yet")

    @component.output_types(answer=str, sources=List[Document])
    def run(self, query: str, documents: List[Document]):
        # Phase 3: build prompt → call LLM → return answer + source docs
        raise NotImplementedError("Phase 3 — not implemented yet")
