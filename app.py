"""

Screens:
- Single Verification: enter a statement, retrieve context, and predict TRUE/FALSE/NONE.

This UI uses the modular backend in the `ramayana` package and caches heavy resources.
"""
from __future__ import annotations

import os
import io
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

import ramayana as rb
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Ramayana Verifier",
    page_icon="📜",
    layout="wide",
)

# -------------------------
# Helpers with Streamlit cache
# -------------------------

@st.cache_data(show_spinner=False)
def load_dataset_cached(dataset_path: str) -> Tuple[pd.DataFrame, list[str]]:
    """Load dataset and return DataFrame + cleaned verses list."""
    df = rb.load_dataset(dataset_path)
    verses = rb.clean_verses(df)
    return df, verses


@st.cache_data(show_spinner=True)
def load_embeddings_and_pca_cached(
    verses: list[str],
    embeddings_path: str,
    st_model_name: str,
    pca_components: int,
) -> Tuple[np.ndarray, rb.PCA, np.ndarray]:
    """Load or compute verse embeddings and fit PCA; return (embs, pca, embs_pca)."""
    # Embeddings are computed via SentenceTransformer inside the backend function.
    # It will load cached .npy if present.
    embs = rb.load_or_compute_embeddings(
        verses=verses,
        embeddings_path=embeddings_path,
        st_model_name=st_model_name,
        device="cpu",
        batch_size=64,
        normalize=False,
    )
    pca, embs_pca = rb.fit_pca(embs, n_components=pca_components)
    return embs, pca, embs_pca


@st.cache_resource(show_spinner=True)
def get_embedding_model_cached(st_model_name: str) -> rb.SentenceTransformer:
    """Cache the SentenceTransformer used for statement encoding (CPU)."""
    return rb.SentenceTransformer(st_model_name, device="cpu")


@st.cache_resource(show_spinner=True)
def get_verifier_pipeline_cached(model_name: str, use_8bit: bool, max_new_tokens: int):
    """Cache the HF pipeline used to classify TRUE/FALSE/NONE."""
    return rb.load_verifier_pipeline(
        model_name=model_name,
        use_8bit=use_8bit,
        max_new_tokens=max_new_tokens,
    )


# -------------------------
# Sidebar: Settings
# -------------------------
with st.sidebar:
    st.header("Settings")

    # Data (Upload only)
    uploaded_dataset = st.file_uploader("Upload dataset CSV", type=["csv"]) 
    selected_text_column = None
    if uploaded_dataset is not None:
        try:
            _tmp_df = pd.read_csv(uploaded_dataset)
        except Exception as e:
            st.error(f"Failed to read uploaded dataset: {e}")
            st.stop()
        columns = list(_tmp_df.columns)
        if not columns:
            st.error("Uploaded CSV has no columns.")
            st.stop()
        default_col = "English_translation" if "English_translation" in columns else columns[0]
        default_idx = columns.index(default_col) if default_col in columns else 0
        selected_text_column = st.selectbox(
            "Text column",
            options=columns,
            index=default_idx,
            help="Choose the column containing the verse text.",
        )
        # Reset file pointer for later reads
        try:
            uploaded_dataset.seek(0)
        except Exception:
            pass

    # Models
    st_model_name = st.text_input(
        "Sentence-Transformer",
        value=rb.DEFAULT_ST_MODEL_NAME,
        help="Encoder used for semantic search (CPU). Default: all-MiniLM-L6-v2",
    )
    model_name = st.text_input(
        "Verifier LLM",
        value=rb.DEFAULT_MODEL_NAME,
        help="Seq2Seq model to output TRUE/FALSE/NONE. You can change this.",
    )

    # Retrieval & LLM
    top_k = st.slider("Top-K Context", min_value=1, max_value=15, value=rb.DEFAULT_TOP_K, step=1)
    pca_components = st.slider("PCA Components", min_value=32, max_value=768, value=rb.DEFAULT_PCA_COMPONENTS, step=16,
                               help="Reduced dimension for similarity search; auto-capped to available dims")
    use_8bit = st.checkbox("Use 8-bit quant (GPU only)", value=True)
    max_new_tokens = st.number_input("Max new tokens", min_value=4, max_value=32, value=8, step=2)

    st.markdown("---")
    # Env info
    st.caption(
        f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}"
        + (f" | CUDA Ver: {torch.version.cuda}" if torch.cuda.is_available() else "")
    )

# -------------------------
# Lazy-load resources
# -------------------------

if uploaded_dataset is None or selected_text_column is None:
    st.warning("Please upload a dataset CSV and choose the text column.")
    st.stop()

with st.spinner("Loading dataset..."):
    df_verses = pd.read_csv(uploaded_dataset)
    if selected_text_column not in df_verses.columns:
        st.error(f"Selected column '{selected_text_column}' not found in the uploaded CSV.")
        st.stop()
    verses = df_verses[selected_text_column].fillna("").astype(str).tolist()

embeddings_path = os.path.join(SCRIPT_DIR, rb.DEFAULT_EMBEDDINGS_PATH)  # created behind the scenes
with st.spinner("Preparing embeddings and PCA..."):
    verse_embs, pca, verse_embs_pca = load_embeddings_and_pca_cached(
        verses=verses,
        embeddings_path=embeddings_path,
        st_model_name=st_model_name,
        pca_components=pca_components,
    )

# Always keep a CPU SentenceTransformer for statements
embedding_model = get_embedding_model_cached(st_model_name)

# Load verifier LLM pipeline
with st.spinner("Loading verifier LLM..."):
    verifier_pipe = get_verifier_pipeline_cached(model_name, use_8bit, max_new_tokens)

# -------------------------
# Header
# -------------------------
st.title("📜 Ramayana Statement Verifier")
st.write("Verify statements using retrieved Ramayana context and an LLM (TRUE/FALSE/NONE).")

# Tabs
single_tab, about_tab = st.tabs(["Single Verification", "About"])

# -------------------------
# Single Verification Tab
# -------------------------
with single_tab:
    from ramayana.ui_single import render_single_tab
    render_single_tab(
        verses=verses,
        embedding_model=embedding_model,
        pca=pca,
        verse_embs_pca=verse_embs_pca,
        verifier_pipe=verifier_pipe,
        top_k=top_k,
        prompt_template=rb.DEFAULT_PROMPT_TEMPLATE,
    )

 

# -------------------------
# About Tab
# -------------------------
with about_tab:
    st.subheader("About")
    st.markdown(
        "- **Dataset**: Uses `cleaned_Ramayana_Dataset.csv` and column `English_translation`.\n"
        "- **Embeddings**: `all-MiniLM-L6-v2` on CPU. Embeddings cached to `verse_embeddings.npy`.\n"
        "- **PCA**: Dimensionality reduction before similarity search.\n"
        "- **LLM**: `google/flan-t5-large` via Transformers pipeline. 8-bit quantization enabled when CUDA is available.\n"
        "- **Output**: Single label among TRUE / FALSE / NONE."
    )

    st.markdown("---")
    st.markdown("### Environment")
    st.write({
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_map": rb.detect_device_map(),
    })

    st.markdown(
        "This app follows the backend structure from `version_2.ipynb`, separating data, embeddings, retrieval, and verification."
    )
