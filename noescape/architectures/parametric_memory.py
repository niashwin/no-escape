"""
Architecture 5: Parametric (Weight-Based) Memory

Memory stored in model weights — no external store.
Probe factual knowledge via Q&A without RAG.
Proximity = cosine similarity of hidden states at middle layers.
d_eff = participation ratio of activations for Wikipedia sentences.
"""

import torch
import numpy as np
import gc
from typing import List, Tuple, Dict, Optional

from noescape.architectures.base import MemoryArchitecture


class ParametricMemoryArchitecture(MemoryArchitecture):
    """Architecture 5: Parametric memory in model weights."""

    @property
    def name(self) -> str:
        return "Parametric Memory (Qwen2.5-7B)"

    @property
    def arch_key(self) -> str:
        return "parametric"

    def _setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        arch_cfg = self.config['architectures']['parametric']
        model_key = arch_cfg.get('model', 'qwen')
        model_cfg = self.config['models'][model_key]
        model_id = model_cfg['hf_id']

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
        self._hidden_dim = self.model.config.hidden_size
        self._items: List[str] = []
        self._metadata: List[dict] = []

    def encode(self, items: List[str]) -> np.ndarray:
        """Get middle-layer hidden states as representations."""
        all_hiddens = []
        for item in items:
            hidden = self._get_hidden(item)
            all_hiddens.append(hidden)
        return np.array(all_hiddens)

    def _get_hidden(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        n_layers = len(outputs.hidden_states)
        mid_layer = n_layers // 2
        hidden = outputs.hidden_states[mid_layer][0].mean(dim=0).cpu().numpy()
        return hidden / (np.linalg.norm(hidden) + 1e-8)

    def store(self, items: List[str], metadata: Optional[List[dict]] = None) -> None:
        """Parametric memory doesn't 'store' externally — items are for reference."""
        self._items.extend(items)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in items])

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        """Generate answer from parametric memory (no RAG)."""
        if not self._items:
            return []

        # Get model's answer
        answer = self.answer_question(query)

        # Score stored items by hidden state similarity to query
        query_hidden = self._get_hidden(query)
        scores = []
        for i, item in enumerate(self._items):
            item_hidden = self._get_hidden(item)
            sim = float(np.dot(query_hidden, item_hidden))
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def answer_question(self, question: str, max_new_tokens: int = 20) -> str:
        """Answer a question using only parametric knowledge."""
        prompt = f"Answer the following question concisely.\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=0.0,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip()
        return answer

    def get_token_probability(self, prompt: str, target_token: str) -> float:
        """Get probability of target token as next token."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
        if target_ids:
            return float(probs[target_ids[0]].cpu())
        return 0.0

    def check_word_in_list(self, word_list: List[str], query_word: str) -> bool:
        """Ask model if a word was in a list (DRM experiment)."""
        list_text = ", ".join(word_list)
        prompt = (
            f"I am going to give you a list of words: [{list_text}]. "
            f"Was the word \"{query_word}\" in the list? Answer yes or no:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=10, temperature=0.0,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip().lower()

        tokens = answer.split()
        if tokens and "yes" in tokens[0]:
            return True
        return False

    def get_similarity(self, item_a: str, item_b: str) -> float:
        emb_a = self._get_hidden(item_a)
        emb_b = self._get_hidden(item_b)
        return float(np.dot(emb_a, emb_b))

    def get_effective_dimensionality(self, items: Optional[List[str]] = None) -> Dict:
        from noescape.utils import compute_participation_ratio
        if items is None:
            items = self._items[:500] if self._items else []
        if not items:
            raise ValueError("No items")
        embeddings = self.encode(items[:500])
        result = compute_participation_ratio(embeddings)
        result['d_nominal'] = self._hidden_dim
        return result

    def clear(self) -> None:
        self._items.clear()
        self._metadata.clear()

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
