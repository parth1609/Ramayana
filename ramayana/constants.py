"""
ramayana/constants.py

Global constants for defaults and labels.
"""
from __future__ import annotations

DEFAULT_DATASET_PATH = "cleaned_Ramayana_Dataset.csv"
DEFAULT_EMBEDDINGS_PATH = "verse_embeddings.npy"
DEFAULT_STATEMENTS_PATH = "initial.csv"

DEFAULT_MODEL_NAME = "google/flan-t5-large"
DEFAULT_ST_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5
DEFAULT_PCA_COMPONENTS = 350

VALID_LABELS = {"TRUE", "FALSE", "NONE"}
