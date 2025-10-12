"""
ramayana/ui_single.py

Streamlit renderer for the Single Verification tab.
"""
from __future__ import annotations

import time
import streamlit as st

import ramayana as rb


def render_single_tab(
    verses,
    embedding_model,
    pca,
    verse_embs_pca,
    verifier_pipe,
    top_k: int,
    prompt_template: str,
):
    """Render the Single Verification tab content."""
    st.subheader("Single Verification")
    default_stmt = "Rama is the eldest son of King Dasharatha."
    statement = st.text_area("Enter a statement to verify", value=default_stmt, height=100)
    run_single = st.button("Verify", type="primary")

    if run_single and statement.strip():
        with st.spinner("Retrieving context and verifying..."):
            start = time.time()
            out = rb.verify_statement(
                statement=statement.strip(),
                verses=verses,
                embedding_model=embedding_model,
                pca=pca,
                verse_embeddings_pca=verse_embs_pca,
                verifier_pipe=verifier_pipe,
                top_k=top_k,
                prompt_template=prompt_template,
                joiner="\n\n",
            )
            elapsed = time.time() - start

        # Display result
        st.markdown(f"**Predicted:** `{out.predicted}` · ⏱️ {elapsed:.2f}s")
        with st.expander("Retrieved Context", expanded=True):
            for i, (ctx, score, idx) in enumerate(zip(out.retrieval.contexts, out.retrieval.scores, out.retrieval.indices)):
                st.markdown(f"**{i+1}. [idx {idx}] score={score:.4f}**\n\n{ctx}")
                st.markdown("---")

        with st.expander("Prompt sent to LLM"):
            st.code(out.prompt)
        with st.expander("Raw model output"):
            st.code(out.raw_generation)
