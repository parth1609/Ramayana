"""
ramayana/types.py

Typed containers for verification workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RetrievalResult:
    """Container for retrieval outputs.

    Attributes:
        indices: List of retrieved verse indices in the dataset.
        scores: Cosine similarity scores corresponding to indices.
        contexts: Retrieved verse texts (cleaned) corresponding to indices.
    """

    indices: List[int]
    scores: List[float]
    contexts: List[str]


@dataclass
class VerificationOutput:
    """Container for verification results.

    Attributes:
        predicted: The predicted label among TRUE/FALSE/NONE.
        context: The concatenated retrieved context string sent to the LLM.
        raw_generation: The raw generated text from the LLM.
        retrieval: The retrieval result with indices/scores/contexts.
        prompt: The full prompt sent to the LLM.
    """

    predicted: str
    context: str
    raw_generation: str
    retrieval: RetrievalResult
    prompt: str
