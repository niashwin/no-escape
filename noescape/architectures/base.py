"""
Base class for all memory architectures.

Every architecture in the No Escape paper must implement this interface.
The experiment code (ebbinghaus.py, drm.py, spacing.py, tot.py) calls
these methods without knowing which architecture is being used, enabling
apples-to-apples comparison.

Architecture implementations:
  - vector_db.py     Architecture 1: Cosine similarity on embeddings (calibration)
  - attention_memory.py  Architecture 2: LLM context window retrieval
  - filesystem_memory.py Architecture 3: JSON files + LLM relevance judge
  - graph_memory.py      Architecture 4: Knowledge graph + PageRank
  - parametric_memory.py Architecture 5: LLM weight-based factual recall
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np


class MemoryArchitecture(ABC):
    """
    Abstract base class for all memory architectures tested in the No Escape paper.

    Each architecture implements a different approach to storing and retrieving
    information, but all must support the same experimental interface so that
    the four experiments (Ebbinghaus, DRM, Spacing, TOT) can run identically
    across all architectures.
    """

    def __init__(self, config: dict):
        """
        Initialize with global config dict (loaded from config.yaml).

        Args:
            config: Full config dict. Architecture-specific params are in
                    config['architectures'][self.arch_key].
        """
        self.config = config
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Architecture-specific initialization (load models, etc.)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name (e.g., 'Vector Database (BGE-large)')."""
        pass

    @property
    @abstractmethod
    def arch_key(self) -> str:
        """Config key (e.g., 'vector_db', 'attention', 'filesystem', 'graph', 'parametric')."""
        pass

    @abstractmethod
    def encode(self, items: List[str]) -> np.ndarray:
        """
        Encode text items into the architecture's representation space.

        For vector/graph architectures, this returns actual embeddings.
        For attention/filesystem/parametric, this may return a proxy
        representation (e.g., hidden states, relevance vectors).

        Args:
            items: List of text strings to encode.

        Returns:
            np.ndarray of shape (len(items), dim) — representations in the
            architecture's native space.
        """
        pass

    @abstractmethod
    def store(self, items: List[str], metadata: Optional[List[dict]] = None) -> None:
        """
        Store items in memory.

        Args:
            items: List of text strings to store.
            metadata: Optional list of metadata dicts (one per item).
                      Common keys: 'timestamp', 'category', 'source',
                      'article_id', 'position'.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        """
        Retrieve top-k items from memory given a query.

        Args:
            query: Query text string.
            top_k: Number of items to retrieve.

        Returns:
            List of (index, score) tuples, sorted by score descending.
            Index refers to the position in the order items were stored.
            Score is the architecture's native similarity/relevance measure
            (higher = more relevant).
        """
        pass

    @abstractmethod
    def get_similarity(self, item_a: str, item_b: str) -> float:
        """
        Compute pairwise similarity between two items under this architecture's
        native proximity measure.

        For vector DB: cosine similarity
        For attention: attention weight (mean over heads)
        For filesystem: LLM relevance score (normalized to [0,1])
        For graph: personalized PageRank score
        For parametric: cosine similarity of hidden states

        Args:
            item_a: First text string.
            item_b: Second text string.

        Returns:
            Similarity score (higher = more similar). Range depends on architecture.
        """
        pass

    @abstractmethod
    def get_effective_dimensionality(self, items: Optional[List[str]] = None) -> Dict:
        """
        Compute effective dimensionality of this architecture's representation space.

        Uses the participation ratio: d_eff = (Σλ_i)² / Σ(λ_i²)
        where λ_i are eigenvalues of the covariance matrix of representations.

        Also reports d_95 and d_99 (components needed for 95%/99% variance).

        Args:
            items: Optional list of items to use for computing dimensionality.
                   If None, uses a default sample from Wikipedia.

        Returns:
            Dict with keys: 'd_eff', 'd_95', 'd_99', 'd_nominal',
                          'eigenvalues' (top 100), 'explained_variance_ratio'.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored memories, reset to empty state."""
        pass

    def get_stored_count(self) -> int:
        """Return number of items currently in memory. Override if needed."""
        raise NotImplementedError

    def get_item_text(self, index: int) -> str:
        """Return the text of the item at the given index. Override if needed."""
        raise NotImplementedError

    # ---- Utility methods (shared across architectures) ----

    def verify_spp(self, related_pairs: List[Tuple[str, str]],
                   unrelated_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Verify the Semantic Proximity Property (SPP) for this architecture.

        Tests whether semantically related items have higher similarity than
        unrelated items under this architecture's proximity measure.

        Args:
            related_pairs: List of (item_a, item_b) pairs that are semantically related.
            unrelated_pairs: List of (item_a, item_b) pairs that are unrelated.

        Returns:
            Dict with 'related_mean', 'unrelated_mean', 't_statistic', 'p_value',
            'spp_satisfied' (True if p < 0.001).
        """
        from scipy import stats

        related_sims = [self.get_similarity(a, b) for a, b in related_pairs]
        unrelated_sims = [self.get_similarity(a, b) for a, b in unrelated_pairs]

        t_stat, p_val = stats.ttest_ind(related_sims, unrelated_sims, alternative='greater')

        return {
            'related_mean': float(np.mean(related_sims)),
            'related_std': float(np.std(related_sims)),
            'unrelated_mean': float(np.mean(unrelated_sims)),
            'unrelated_std': float(np.std(unrelated_sims)),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'spp_satisfied': bool(p_val < 0.001),
            'n_related': len(related_pairs),
            'n_unrelated': len(unrelated_pairs),
        }
