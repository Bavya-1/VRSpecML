"""SBERT sentence encoder (all-MiniLM-L6-v2, as in ../mapper) — retrieval only, not generation."""
import os
import threading
from typing import List

import numpy as np

import config

_lock = threading.Lock()
_encode_lock = threading.Lock()
_model = None


def get_model():
    global _model
    with _lock:
        if _model is None:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            snapshots = os.path.join(config.MODELS_DIR, "models--" + config.EMBED_MODEL.replace("/", "--"), "snapshots")
            if os.path.isdir(snapshots) and os.listdir(snapshots):
                os.environ.setdefault("HF_HUB_OFFLINE", "1")  # cached: no network round-trips
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(config.EMBED_MODEL, cache_folder=config.MODELS_DIR, device="cpu")
        return _model


def embed(texts: List[str], batch_size: int = config.EMBED_BATCH, progress=None) -> np.ndarray:
    """L2-normalised float32 embeddings (cosine similarity == inner product)."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_model()
    out = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        with _encode_lock:
            vecs = model.encode(batch, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        out.append(vecs.astype("float32"))
        if progress:
            progress(min(start + batch_size, total), total)
    return np.vstack(out)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
