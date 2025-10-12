"""
ramayana/ui_batch.py

Streamlit renderer for the Batch Evaluation tab.
"""
from __future__ import annotations

from typing import Optional
import os

import pandas as pd
import streamlit as st

import ramayana as rb


def render_batch_tab(
    verses,
    embedding_model,
    pca,
    verse_embs_pca,
    verifier_pipe,
    default_statements_path: str,
    top_k_default: int,
    prompt_template: str,
):
    """Render the Batch Evaluation tab content."""
    st.subheader("Batch Evaluation")
    st.write("Use an input CSV with columns `Statement` and `Truth`. Defaults to `initial.csv`.")

    col1, col2 = st.columns([1, 1])
    with col1:
        use_default = st.checkbox("Use repository initial.csv", value=True)
        uploaded = None
        if not use_default:
            uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
    with col2:
        batch_top_k = st.slider("Top-K for batch", 1, 15, top_k_default)

    df_input: Optional[pd.DataFrame] = None
    if use_default:
        default_path = default_statements_path
        if not os.path.exists(default_path):
            st.warning(f"Default statements CSV not found at {default_path}")
        else:
            df_input = pd.read_csv(default_path)
    elif uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

    if df_input is not None:
        st.dataframe(df_input.head(10), use_container_width=True)
        run_batch = st.button("Run Evaluation", type="primary")
        if run_batch:
            progress = st.progress(0.0)
            status = st.empty()

            def on_progress(done: int, total: int):
                frac = done / max(total, 1)
                progress.progress(frac)
                status.text(f"Processed {done}/{total}")

            with st.spinner("Running batch verification..."):
                df_out = rb.batch_verify(
                    df_statements=df_input,
                    verses=verses,
                    embedding_model=embedding_model,
                    pca=pca,
                    verse_embeddings_pca=verse_embs_pca,
                    verifier_pipe=verifier_pipe,
                    top_k=batch_top_k,
                    prompt_template=prompt_template,
                    joiner="\n\n",
                    progress_cb=on_progress,
                )
            status.text("Done")

            # Compute accuracy if labels available
            try:
                truth = df_out["Actual_Truth"].str.upper()
                pred = df_out["Predicted_Truth"].str.upper()
                acc = (truth == pred).mean()
                st.markdown(f"**Accuracy:** {acc:.2%}")
            except Exception:
                pass

            st.dataframe(df_out, use_container_width=True)

            # Download link
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                data=csv_bytes,
                file_name="Statements_predictions.csv",
                mime="text/csv",
            )
