"""
ramayana/verification.py

Verification pipeline: label parsing, single-statement verify, and batch verify.
"""
from __future__ import annotations

from typing import Optional, Any

import pandas as pd

from .constants import VALID_LABELS, DEFAULT_TOP_K
from .types import VerificationOutput
from .retrieval import retrieve_top_k
from .prompts import build_prompt, DEFAULT_PROMPT_TEMPLATE


def parse_label(text: str) -> str:
    """Parse the model output to one of TRUE/FALSE/NONE.

    Parameters:
        text: Raw generated string from the LLM.

    Returns:
        Uppercase label among {TRUE, FALSE, NONE}. Defaults to 'NONE' if unclear.
    """
    t = (text or "").strip().upper()
    if t in VALID_LABELS:
        return t
    for lab in ("TRUE", "FALSE", "NONE"):
        if lab in t:
            return lab
    return "NONE"


def verify_statement(
    statement: str,
    verses,
    embedding_model,
    pca,
    verse_embeddings_pca,
    verifier_pipe,
    top_k: int = DEFAULT_TOP_K,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    joiner: str = "\n\n",
    generation_kwargs: Optional[dict[str, Any]] = None,
) -> VerificationOutput:
    """Run retrieval-augmented verification for a single statement.

    Returns:
        VerificationOutput with predicted label, context, raw text, retrieval info, and prompt.
    """
    rr = retrieve_top_k(
        statement=statement,
        embedding_model=embedding_model,
        pca=pca,
        verse_embeddings_pca=verse_embeddings_pca,
        verses=verses,
        top_k=top_k,
    )

    context_str = joiner.join(rr.contexts)
    prompt = build_prompt(statement, context_str, template=prompt_template)

    gen_args = {"max_new_tokens": 8}
    if generation_kwargs:
        gen_args.update(generation_kwargs)

    outputs = verifier_pipe(prompt, **gen_args)
    raw_text = outputs[0].get("generated_text", "") if outputs else ""
    label = parse_label(raw_text)

    return VerificationOutput(
        predicted=label,
        context=context_str,
        raw_generation=raw_text,
        retrieval=rr,
        prompt=prompt,
    )


def batch_verify(
    df_statements: pd.DataFrame,
    verses,
    embedding_model,
    pca,
    verse_embeddings_pca,
    verifier_pipe,
    top_k: int = DEFAULT_TOP_K,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    joiner: str = "\n\n",
    progress_cb: Optional[Any] = None,
) -> pd.DataFrame:
    """Batch verify a DataFrame of statements. Expects columns ['Statement','Truth']."""
    cols_needed = {"Statement", "Truth"}
    if not cols_needed.issubset(set(df_statements.columns)):
        raise ValueError("df_statements must contain ['Statement','Truth'] columns")

    results: list[dict[str, str]] = []
    total = len(df_statements)

    for _, row in df_statements.iterrows():
        stmt = str(row["Statement"]) if pd.notna(row["Statement"]) else ""
        truth = str(row["Truth"]).upper() if pd.notna(row["Truth"]) else "NONE"
        try:
            out = verify_statement(
                statement=stmt,
                verses=verses,
                embedding_model=embedding_model,
                pca=pca,
                verse_embeddings_pca=verse_embeddings_pca,
                verifier_pipe=verifier_pipe,
                top_k=top_k,
                prompt_template=prompt_template,
                joiner=joiner,
            )
            pred = out.predicted
            context = out.context
        except Exception as e:
            pred = "NONE"
            context = f"Error: {e}"

        results.append(
            {
                "Original_Statement": stmt,
                "Actual_Truth": truth,
                "Predicted_Truth": pred,
                "Retrieved_Context": context,
                "Match": str(truth == pred),
                "Reasoning": "",
            }
        )

        if progress_cb:
            try:
                progress_cb(len(results), total)
            except Exception:
                pass

    return pd.DataFrame(results)
