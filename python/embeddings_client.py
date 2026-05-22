import os
from typing import Sequence

import numpy as np
import requests

from config.settings import EMBEDDING_URL, EMBED_BATCH_SIZE, EMBED_REQUEST_TIMEOUT


class EmbeddingsClient:
    def __init__(self, base_url: str | None = None, batch_size: int = EMBED_BATCH_SIZE):
        self.base_url = (base_url or os.getenv("EMBEDDING_URL", EMBEDDING_URL)).rstrip("/")
        self.batch_size = batch_size

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        payload = {"inputs": texts}
        response = requests.post(
            f"{self.base_url}/",
            json=payload,
            timeout=EMBED_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)
        parts = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            print(f"embeddings batch {start // self.batch_size + 1}, size={len(batch)}")
            parts.append(self._embed_batch(batch))
        return np.vstack(parts)
