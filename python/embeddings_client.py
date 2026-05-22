import os
from typing import Sequence

import numpy as np
import requests

from config.settings import EMBEDDING_URL


class EmbeddingsClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("EMBEDDING_URL", EMBEDDING_URL)).rstrip("/")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        payload = {"inputs": list(texts)}
        response = requests.post(f"{self.base_url}/", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms
