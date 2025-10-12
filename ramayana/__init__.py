"""
ramayana package

Modular backend for the Ramayana RAG-style verification pipeline.
This package exposes a stable facade to be used by the Streamlit app.
"""
from __future__ import annotations

# Re-export common types and utilities
from sklearn.decomposition import PCA  # for external annotations
from sentence_transformers import SentenceTransformer  # for type hints & convenience

from .constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_STATEMENTS_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_ST_MODEL_NAME,
    DEFAULT_TOP_K,
    DEFAULT_PCA_COMPONENTS,
    VALID_LABELS,
)
from .data import load_dataset, clean_verses
from .embeddings import load_or_compute_embeddings, fit_pca
from .retrieval import retrieve_top_k
from .prompts import DEFAULT_PROMPT_TEMPLATE, build_prompt
from .llm import detect_device_map, build_quant_config_if_applicable, load_verifier_pipeline
from .verification import parse_label, verify_statement, batch_verify
from .types import RetrievalResult, VerificationOutput

__all__ = [
    # constants
    "DEFAULT_DATASET_PATH",
    "DEFAULT_EMBEDDINGS_PATH",
    "DEFAULT_STATEMENTS_PATH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_ST_MODEL_NAME",
    "DEFAULT_TOP_K",
    "DEFAULT_PCA_COMPONENTS",
    "VALID_LABELS",
    # sklearn export
    "PCA",
    "SentenceTransformer",
    # data
    "load_dataset",
    "clean_verses",
    # embeddings
    "load_or_compute_embeddings",
    "fit_pca",
    # retrieval
    "retrieve_top_k",
    # prompts
    "DEFAULT_PROMPT_TEMPLATE",
    "build_prompt",
    # llm
    "detect_device_map",
    "build_quant_config_if_applicable",
    "load_verifier_pipeline",
    # verification
    "parse_label",
    "verify_statement",
    "batch_verify",
    # types
    "RetrievalResult",
    "VerificationOutput",
]
