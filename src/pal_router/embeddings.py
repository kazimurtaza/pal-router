"""Query embedding for routing classification."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

# Model options (trade-off: speed vs accuracy)
MODELS = {
    "fast": "all-MiniLM-L6-v2",      # 22M params, 80MB, 14k queries/sec
    "balanced": "all-mpnet-base-v2",  # 110M params, 420MB, 2.8k queries/sec
    "accurate": "instructor-base",    # 110M params, instruction-tuned
}

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedder(
    model_name: Literal["fast", "balanced", "accurate"] = "fast"
) -> SentenceTransformer:
    """Get or create a cached sentence transformer model."""
    if model_name not in _model_cache:
        model_id = MODELS[model_name]
        _model_cache[model_name] = SentenceTransformer(model_id)
    return _model_cache[model_name]


def embed_query(query: str, model_name: str = "fast") -> np.ndarray:
    """Embed a single query."""
    embedder = get_embedder(model_name)
    return embedder.encode(query, convert_to_numpy=True)


def embed_queries(queries: list[str], model_name: str = "fast") -> np.ndarray:
    """Embed multiple queries (batched for efficiency)."""
    embedder = get_embedder(model_name)
    return embedder.encode(queries, convert_to_numpy=True, show_progress_bar=True)
