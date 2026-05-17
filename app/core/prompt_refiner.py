"""
1st stage: Translator & Query Refiner
ChatPromptBuilder → LLM → refined query (text) → embedder
"""

from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage


translator_refiner_template = [ChatMessage.from_user(
    """\
You are an expert automotive engineering query optimizer. Your output will be directly embedded into a Vector Database (VDB) to perform a semantic search against official automotive factory repair manuals (specifically Chery M11).

Your goal is to hunt for and extract the core concepts from the user's query, and format them to exactly match the formal engineering language used by automotive manufacturers.

Instructions:
1. Translate: If the query is not in English, translate it to English first.
2. Normalize Terminology: Map any colloquial, alternative, or varying technical terms (even if technically correct) to the strict, standard engineering vocabulary found in official OEM service manuals.
3. Optimize for VDB: Extract only the high-value keywords and precise technical phrases that will yield the best semantic similarity matches in a vector search.
4. Clean: Remove all conversational filler, question words, and irrelevant context.
5. Retain Specifics: Keep all exact specifications, DTCs, part numbers, and units (e.g., torque, N·m, mm).

Query: {{ query }}

Return ONLY the final, refined English keywords for embedding. No explanations.\
"""
)]

