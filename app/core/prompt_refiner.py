"""
1st stage: Translator & Query Refiner
ChatPromptBuilder → LLM → refined query (text) → embedder
"""

from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_ai import GoogleGenAIChatGenerator


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

refiner_prompt_builder = ChatPromptBuilder(template=translator_refiner_template)

gemini_refiner_agent = GoogleGenAIChatGenerator(
    model="gemini-2.0-flash",
    generation_config={"temperature": 0.2},
)
