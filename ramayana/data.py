"""
ramayana/data.py

Data loading and preprocessing utilities.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from .constants import DEFAULT_DATASET_PATH


def load_dataset(dataset_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the Ramayana verses dataset.

    Parameters:
        dataset_path: Path to the CSV file containing verses.

    Returns:
        A pandas DataFrame with column 'English_translation'.

    Side Effects:
        None.

    Examples:
        >>> df = load_dataset("cleaned_Ramayana_Dataset.csv")
        >>> 'English_translation' in df.columns
        True
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    if "English_translation" not in df.columns:
        raise ValueError("Dataset must contain an 'English_translation' column")
    return df


def clean_verses(df: pd.DataFrame) -> List[str]:
    """Clean verse texts: replace None/NaN with empty strings and ensure str type.

    Parameters:
        df: DataFrame with column 'English_translation'.

    Returns:
        A list of cleaned verse strings, aligned with df rows.
    """
    cleaned: List[str] = []
    for v in df["English_translation"].tolist():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            cleaned.append("")
        else:
            cleaned.append(str(v))
    return cleaned
