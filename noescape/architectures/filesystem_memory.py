"""
Architecture 3: Filesystem-Based Agent Memory

Storage: each memory as a JSON record with content, timestamp, tags, summary.
Retrieval: BM25 keyword search -> top candidates -> LLM (Qwen) selects most relevant.
Proximity: LLM relevance score (1-10 scale, normalized to [0,1]).
"""

import torch
import numpy as np
import gc
import json
from typing import List, Tuple, Dict, Optional
from rank_bm25 import BM25Okapi

from noescape.architectures.base import MemoryArchitecture


class FilesystemMemoryArchitecture(MemoryArchitecture):
    """Architecture 3: Filesystem/agent memory with BM25 + LLM re-ranking."""

    @property
    def name(self) -> str:
        return "Filesystem Memory (BM25 + Qwen)"

    @property
    def arch_key(self) -> str:
        return "filesystem"

    def _setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        arch_cfg = self.config['architectures']['filesystem']
        model_key = arch_cfg.get('judge_model', 'qwen')
        model_cfg = self.config['models'][model_key]
        model_id = model_cfg['hf_id']

        self.bm25_top_k = arch_cfg.get('bm25_top_k', 50)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=quant_config,
                device_map="auto", trust_remote_code=True,
            )
        except Exception:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )
            except Exception:
                fallback_id = model_cfg.get('fallback', 'Qwen/Qwen2.5-3B-Instruct')
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_id, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )

        self.model.eval()
        self._items: List[str] = []
        self._metadata: List[dict] = []
        self._bm25 = None
        self._tokenized_corpus: List[List[str]] = []

        # Also load embedding model for encode() / dimensionality
        import sys
        from pathlib import Path
        HIDE_ROOT = Path(__file__).parent.parent.parent / "hide-project"
        sys.path.insert(0, str(HIDE_ROOT))
        from hide.models.embedding_models import EmbeddingManager
        self.embedding_manager = EmbeddingManager(model_name="bge-large", device="cuda:0")
        self.embedding_manager.load()
        self._dim = self.embedding_manager.dim

    def _rebuild_bm25(self):
        """Rebuild BM25 index from current corpus."""
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    def encode(self, items: List[str]) -> np.ndarray:
        """Use BGE-large embeddings as proxy representation."""
        return self.embedding_manager.encode(items, batch_size=256)

    def store(self, items: List[str], metadata: Optional[List[dict]] = None) -> None:
        self._items.extend(items)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in items])
        for item in items:
            self._tokenized_corpus.append(item.lower().split())
        self._rebuild_bm25()

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        if not self._items or self._bm25 is None:
            return []

        # BM25 first pass
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        n_candidates = min(self.bm25_top_k, len(self._items))
        candidate_indices = np.argsort(bm25_scores)[::-1][:n_candidates]

        # LLM re-ranking for top candidates
        scored = []
        for idx in candidate_indices:
            score = self._llm_relevance_score(query, self._items[idx])
            scored.append((int(idx), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _llm_relevance_score(self, query: str, memory_content: str) -> float:
        """Get LLM relevance score (1-10), normalized to [0,1]."""
        prompt = (
            f"Rate the relevance of this memory to the query on a scale of 1-10.\n"
            f"Query: {query}\n"
            f"Memory: {memory_content}\n"
            f"Relevance (1-10):"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=5, temperature=0.0,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip()
        try:
            score = int(''.join(c for c in answer if c.isdigit())[:2])
            score = max(1, min(10, score))
        except (ValueError, IndexError):
            score = 5
        return score / 10.0

    def get_similarity(self, item_a: str, item_b: str) -> float:
        """Use LLM relevance score as similarity."""
        return self._llm_relevance_score(item_a, item_b)

    def get_effective_dimensionality(self, items: Optional[List[str]] = None) -> Dict:
        from noescape.utils import compute_participation_ratio
        if items is not None:
            embeddings = self.encode(items)
        elif self._items:
            embeddings = self.encode(self._items[:1000])
        else:
            raise ValueError("No items")
        result = compute_participation_ratio(embeddings)
        result['d_nominal'] = self._dim
        return result

    def clear(self) -> None:
        self._items.clear()
        self._metadata.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None

    def get_stored_count(self) -> int:
        return len(self._items)

    def get_item_text(self, index: int) -> str:
        return self._items[index]

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
