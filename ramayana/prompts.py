"""
ramayana/prompts.py

Prompt templates and rendering.
"""
from __future__ import annotations

DEFAULT_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Based only and directly on the provided Ramayana Context, "
    "determine if the Statement is TRUE, FALSE, or NONE.\n"
    "Output only the single word TRUE, FALSE, or NONE.\n\n"
    "Definitions:\n"
    "- TRUE — explicitly and unambiguously confirmed by the context.\n"
    "- FALSE — explicitly contradicted or logically impossible based on the context.\n"
    "- NONE — not covered, too vague, or unrelated to the Ramayana context provided.\n\n"
    "Rules:\n"
    "1. If a statement is partially true or contains errors based on the context, answer FALSE.\n"
    "2. If the statement requires assumptions beyond the provided context or is subjective, answer NONE.\n"
    "3. Use only literal matches from the context; do not infer.\n\n"
    "Now, based on the following context and statement:\n"
    "Context: \"{context}\"\n"
    "Statement: {stmt}\n"
    "Answer:"
)


def build_prompt(statement: str, context: str, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    """Render the verification prompt using the provided template."""
    return template.format(context=context, stmt=statement)
