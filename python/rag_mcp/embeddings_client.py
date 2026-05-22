import os

import httpx
import numpy as np

from rag_mcp.config import CONFIG


class EmbeddingsClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or CONFIG["embedding_url"]
        self.model = model or CONFIG["embedding_model"]
        self.batch_size = CONFIG["embedding_batch_size"]

    def embed(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, CONFIG["embedding_dim"]), dtype=np.float32)
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {"model": self.model, "input": batch}
            response = httpx.post(self.base_url, json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()["data"]
            ordered = sorted(data, key=lambda item: item["index"])
            vectors.extend([item["embedding"] for item in ordered])
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return arr / norms
