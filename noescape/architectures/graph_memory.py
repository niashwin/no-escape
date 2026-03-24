"""
Architecture 4: Graph-Based Memory

Nodes are sentence embeddings (MiniLM or BGE), edges connect if cosine > threshold.
Retrieval via personalized PageRank from query node.
d_eff computed from graph adjacency matrix eigenvalues.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional

from noescape.architectures.base import MemoryArchitecture


class GraphMemoryArchitecture(MemoryArchitecture):
    """Architecture 4: Graph-based memory with PageRank retrieval."""

    @property
    def name(self) -> str:
        return "Graph Memory (MiniLM + PageRank)"

    @property
    def arch_key(self) -> str:
        return "graph"

    def _setup(self):
        import sys
        from pathlib import Path
        HIDE_ROOT = Path(__file__).parent.parent.parent / "hide-project"
        sys.path.insert(0, str(HIDE_ROOT))
        from hide.models.embedding_models import EmbeddingManager

        arch_cfg = self.config['architectures']['graph']
        model_key = arch_cfg.get('model', 'minilm')
        self.edge_threshold = arch_cfg.get('edge_threshold', 0.7)
        self.pagerank_alpha = arch_cfg.get('pagerank_alpha', 0.85)

        self.embedding_manager = EmbeddingManager(
            model_name=model_key,
            device="cuda:0"
        )
        self.embedding_manager.load()
        self._dim = self.embedding_manager.dim

        self.graph = nx.Graph()
        self._items: List[str] = []
        self._embeddings = np.zeros((0, self._dim), dtype=np.float32)
        self._metadata: List[dict] = []

    def encode(self, items: List[str]) -> np.ndarray:
        return self.embedding_manager.encode(items, batch_size=256)

    def store(self, items: List[str], metadata: Optional[List[dict]] = None) -> None:
        embeddings = self.encode(items)
        start_idx = len(self._items)

        if self._embeddings.shape[0] == 0:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        self._items.extend(items)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in items])

        # Add nodes and edges
        for i, emb in enumerate(embeddings):
            node_id = start_idx + i
            self.graph.add_node(node_id)

            # Connect to existing nodes if cosine > threshold
            if self._embeddings.shape[0] > 1:
                sims = self._embeddings[:node_id] @ emb
                for j in np.where(sims > self.edge_threshold)[0]:
                    self.graph.add_edge(node_id, int(j), weight=float(sims[j]))

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        if len(self._items) == 0:
            return []

        query_emb = self.encode([query])[0]
        sims = self._embeddings @ query_emb

        # Find closest node as personalization seed
        seed_node = int(np.argmax(sims))

        # Personalized PageRank
        personalization = {seed_node: 1.0}
        try:
            pr = nx.pagerank(self.graph, alpha=self.pagerank_alpha,
                           personalization=personalization, max_iter=100)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            # Fallback to cosine similarity
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [(int(i), float(sims[i])) for i in top_idx]

        # Combine PageRank with cosine similarity
        scores = np.zeros(len(self._items))
        for node, pr_score in pr.items():
            scores[node] = 0.5 * sims[node] + 0.5 * pr_score * len(self._items)

        k = min(top_k, len(self._items))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(int(i), float(scores[i])) for i in top_idx]

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
            raise ValueError("No items for dimensionality computation")

        result = compute_participation_ratio(embeddings)
        result['d_nominal'] = self._dim

        # Also compute graph Laplacian eigenvalues for d_eff
        if self.graph.number_of_nodes() > 10:
            try:
                L = nx.laplacian_matrix(self.graph).toarray().astype(np.float64)
                eigs = np.linalg.eigvalsh(L)
                eigs = np.sort(eigs[eigs > 1e-10])[::-1]
                if len(eigs) > 0:
                    graph_d_eff = float((eigs.sum())**2 / (eigs**2).sum())
                    result['graph_laplacian_d_eff'] = graph_d_eff
            except Exception:
                pass

        return result

    def clear(self) -> None:
        self.graph = nx.Graph()
        self._items.clear()
        self._embeddings = np.zeros((0, self._dim), dtype=np.float32)
        self._metadata.clear()

    def get_stored_count(self) -> int:
        return len(self._items)

    def get_item_text(self, index: int) -> str:
        return self._items[index]
