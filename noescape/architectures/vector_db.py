"""
Architecture 1: Vector Database Retrieval (Calibration Baseline)

Wraps the HIDE space implementation with the standard MemoryArchitecture interface.
Uses BAAI/bge-large-en-v1.5 (1024-dim) with cosine similarity retrieval.
Must reproduce HIDE results within 2 SE for calibration.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add HIDE project to path
HIDE_ROOT = Path(__file__).parent.parent.parent / "hide-project"
sys.path.insert(0, str(HIDE_ROOT))

from noescape.architectures.base import MemoryArchitecture


class VectorDBArchitecture(MemoryArchitecture):
    """Architecture 1: Vector database with cosine similarity."""

    @property
    def name(self) -> str:
        return "Vector Database (BGE-large)"

    @property
    def arch_key(self) -> str:
        return "vector_db"

    def _setup(self):
        from hide.models.embedding_models import EmbeddingManager
        arch_cfg = self.config['architectures']['vector_db']
        model_key = arch_cfg['model']
        model_name = "bge-large" if model_key == "bge_large" else model_key

        self.embedding_manager = EmbeddingManager(
            model_name=model_name,
            device="cuda:0"
        )
        self.embedding_manager.load()
        self._dim = self.embedding_manager.dim

        # Storage
        self._embeddings = np.zeros((0, self._dim), dtype=np.float32)
        self._items: List[str] = []
        self._metadata: List[dict] = []

    def encode(self, items: List[str]) -> np.ndarray:
        return self.embedding_manager.encode(items, batch_size=256)

    def store(self, items: List[str], metadata: Optional[List[dict]] = None) -> None:
        embeddings = self.encode(items)
        if self._embeddings.shape[0] == 0:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        self._items.extend(items)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in items])

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        if self._embeddings.shape[0] == 0:
            return []
        query_emb = self.encode([query])[0]
        similarities = self._embeddings @ query_emb
        k = min(top_k, len(similarities))
        top_idx = np.argpartition(similarities, -k)[-k:]
        top_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]
        return [(int(i), float(similarities[i])) for i in top_idx]

    def get_similarity(self, item_a: str, item_b: str) -> float:
        embs = self.encode([item_a, item_b])
        return float(np.dot(embs[0], embs[1]))

    def get_effective_dimensionality(self, items: Optional[List[str]] = None) -> Dict:
        from noescape.utils import compute_participation_ratio
        if items is not None:
            embeddings = self.encode(items)
        elif self._embeddings.shape[0] > 0:
            embeddings = self._embeddings
        else:
            raise ValueError("No items to compute dimensionality from")
        result = compute_participation_ratio(embeddings)
        result['d_nominal'] = self._dim
        return result

    def clear(self) -> None:
        self._embeddings = np.zeros((0, self._dim), dtype=np.float32)
        self._items.clear()
        self._metadata.clear()

    def get_stored_count(self) -> int:
        return len(self._items)

    def get_item_text(self, index: int) -> str:
        return self._items[index]
