"""
ramayana/llm.py

LLM utilities: device detection, optional 8-bit quantization, and pipeline loading.
"""
from __future__ import annotations

from typing import Optional, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline as hf_pipeline,
)

# Optional import for bitsandbytes config
try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore


def detect_device_map() -> str:
    """Return a device map hint for HF model loading.

    Returns:
        'auto' if CUDA available, else 'cpu'. This string can be used for device_map.
    """
    return "auto" if torch.cuda.is_available() else "cpu"


def build_quant_config_if_applicable() -> Optional[Any]:
    """Create BitsAndBytes 8-bit quantization config only when CUDA and bitsandbytes are available."""
    if not torch.cuda.is_available() or BitsAndBytesConfig is None:
        return None
    try:  # ensure bitsandbytes is importable on this system
        import bitsandbytes as bnb  # noqa: F401
    except Exception:
        return None
    return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)


def load_verifier_pipeline(
    model_name: str,
    use_8bit: bool = True,
    max_new_tokens: int = 8,
):
    """Load the HF text2text-generation pipeline for verification.

    Parameters:
        model_name: Hugging Face model to load (Seq2Seq).
        use_8bit: If True and GPU is available, load with 8-bit quantization.
        max_new_tokens: Default generation length for single-token labels.

    Returns:
        A Hugging Face pipeline callable for text2text-generation.
    """
    device_map = detect_device_map()
    quant_config = build_quant_config_if_applicable() if use_8bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if quant_config is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quant_config,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
        )

    pipe = hf_pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)
    pipe.model.generation_config.max_new_tokens = max_new_tokens
    return pipe
