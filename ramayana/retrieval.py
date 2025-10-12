"""
ramayana/retrieval.py

Context retrieval using cosine similarity on PCA-reduced embeddings.
"""
from __future__ import annotations

from typing import List

import torch
from sentence_transformers import SentenceTransformer, util

from .types import RetrievalResult


def retrieve_top_k(
    statement: str,
    embedding_model: SentenceTransformer,
    pca,
    verse_embeddings_pca,
    verses: List[str],
    top_k: int,
) -> RetrievalResult:
    """Retrieve top-K most similar verses to the given statement.

    Parameters:
        statement: The input statement to verify.
        embedding_model: SentenceTransformer used to embed the statement.
        pca: Fitted PCA used to transform embeddings for similarity.
        verse_embeddings_pca: Verse embeddings already PCA-transformed (np.ndarray).
        verses: The verse texts list.
        top_k: Number of context items to retrieve.

    Returns:
        RetrievalResult with indices, cosine scores, and verse contexts.
    """
    # Encode statement and transform
    stmt_emb = embedding_model.encode(statement)
    stmt_emb_2d = stmt_emb.reshape(1, -1)
    stmt_emb_pca = pca.transform(stmt_emb_2d)

    # Cosine similarity in torch on CPU for consistency
    stmt_tensor = torch.tensor(stmt_emb_pca, device="cpu")
    verses_tensor = torch.tensor(verse_embeddings_pca, device="cpu")
    cos_scores = util.cos_sim(stmt_tensor, verses_tensor)[0]

    k = min(max(top_k, 1), len(verses))
    topk = torch.topk(cos_scores, k=k)
    indices = topk.indices.cpu().tolist()
    scores = topk.values.cpu().tolist()

    contexts = [verses[i] for i in indices]
    return RetrievalResult(indices=indices, scores=scores, contexts=contexts)
