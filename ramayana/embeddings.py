"""
ramayana/embeddings.py

Sentence embeddings and PCA utilities.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


def load_or_compute_embeddings(
    verses: List[str],
    embeddings_path: str,
    st_model_name: str,
    device: str = "cpu",
    batch_size: int = 64,
    normalize: bool = False,
) -> np.ndarray:
    """Load precomputed verse embeddings or compute them using SentenceTransformer.

    Parameters:
        verses: The list of verse texts.
        embeddings_path: Path to a .npy file to load/save embeddings.
        st_model_name: SentenceTransformer model name.
        device: Device for the embedding model (e.g., 'cpu').
        batch_size: Encode batch size.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        NumPy array of shape (num_verses, dim) with embeddings.
    """
    embs: Optional[np.ndarray] = None
    # Try to load cached embeddings
    try:
        import os
        if os.path.exists(embeddings_path):
            loaded = np.load(embeddings_path)
            if loaded.shape[0] == len(verses):
                embs = loaded
    except Exception:
        embs = None

    if embs is None:
        model = SentenceTransformer(st_model_name, device=device)
        embs = model.encode(verses, batch_size=batch_size, show_progress_bar=True)
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
        # Persist
        try:
            np.save(embeddings_path, embs)
        except Exception:
            pass
    return embs


def fit_pca(
    embeddings_np: np.ndarray,
    n_components: int,
) -> Tuple[PCA, np.ndarray]:
    """Fit PCA and transform verse embeddings for similarity search."""
    n_samples, n_features = embeddings_np.shape
    n_comp = min(n_components, n_samples, n_features)
    if n_comp < 1:
        raise ValueError("Invalid PCA components computed from data shape")
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(embeddings_np)
    return pca, transformed
