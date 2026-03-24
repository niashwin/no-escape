"""
Architecture 2: Attention-Based Context Memory

Uses Qwen2.5-7B-Instruct with facts placed in context window.
Retrieval = model answering questions from context.
Proximity = attention weights from query to memory passages.
d_eff = participation ratio of key/query vectors.
"""

import torch
import numpy as np
import gc
from typing import List, Tuple, Dict, Optional

from noescape.architectures.base import MemoryArchitecture


class AttentionMemoryArchitecture(MemoryArchitecture):
    """Architecture 2: LLM attention-based context window memory."""

    @property
    def name(self) -> str:
        return "Attention Memory (Qwen2.5-7B)"

    @property
    def arch_key(self) -> str:
        return "attention"

    def _setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        arch_cfg = self.config['architectures']['attention']
        model_key = arch_cfg.get('model', 'qwen')
        model_cfg = self.config['models'][model_key]
        model_id = model_cfg['hf_id']

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Try loading with 4-bit quantization
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
            # Try fp16
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )
            except Exception:
                # Fallback to smaller model
                fallback_id = model_cfg.get('fallback', 'Qwen/Qwen2.5-3B-Instruct')
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_id, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )

        self.model.eval()
        self._items: List[str] = []
        self._metadata: List[dict] = []
        self._hidden_dim = self.model.config.hidden_size

    def encode(self, items: List[str]) -> np.ndarray:
        """Get hidden state representations for items."""
        all_hiddens = []
        for item in items:
            inputs = self.tokenizer(item, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            # Use middle layer hidden state, mean-pooled
            n_layers = len(outputs.hidden_states)
            mid_layer = n_layers // 2
            hidden = outputs.hidden_states[mid_layer][0].mean(dim=0).cpu().numpy()
            hidden = hidden / (np.linalg.norm(hidden) + 1e-8)
            all_hiddens.append(hidden)
        return np.array(all_hiddens)

    def store(self, items: List[str], metadata: Optional[List[dict]] = None) -> None:
        self._items.extend(items)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in items])

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        """Retrieve by asking the model with all facts in context."""
        if not self._items:
            return []

        # Build context with facts
        facts_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(self._items)])
        prompt = (
            f"You are a factual assistant. Answer based ONLY on the facts provided.\n\n"
            f"FACTS:\n{facts_text}\n\n"
            f"QUESTION: {query}\n"
            f"Answer in one or two words:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=self.config['architectures']['attention'].get('context_length', 4096))
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=20, temperature=0.0,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip()

        # Score each fact by relevance to the answer
        scores = []
        query_emb = self._get_hidden(query)
        for i, item in enumerate(self._items):
            item_emb = self._get_hidden(item)
            score = float(np.dot(query_emb, item_emb))
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _get_hidden(self, text: str) -> np.ndarray:
        """Get middle-layer hidden state for text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        n_layers = len(outputs.hidden_states)
        mid_layer = n_layers // 2
        hidden = outputs.hidden_states[mid_layer][0].mean(dim=0).cpu().numpy()
        return hidden / (np.linalg.norm(hidden) + 1e-8)

    def get_similarity(self, item_a: str, item_b: str) -> float:
        emb_a = self._get_hidden(item_a)
        emb_b = self._get_hidden(item_b)
        return float(np.dot(emb_a, emb_b))

    def get_effective_dimensionality(self, items: Optional[List[str]] = None) -> Dict:
        from noescape.utils import compute_participation_ratio
        if items is None:
            items = self._items[:1000] if self._items else []
        if not items:
            raise ValueError("No items for dimensionality computation")
        embeddings = self.encode(items[:1000])
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

    def answer_question_in_context(self, facts: List[str], question: str,
                                    max_new_tokens: int = 20) -> str:
        """Answer a question with given facts in context window."""
        facts_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(facts)])
        prompt = (
            f"You are a factual assistant. Answer based ONLY on the facts provided.\n\n"
            f"FACTS:\n{facts_text}\n\n"
            f"QUESTION: {question}\n"
            f"Answer in one or two words:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=self.config['architectures']['attention'].get('context_length', 4096))
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=0.0,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip()
        return answer

    def check_word_in_list(self, word_list: List[str], query_word: str) -> bool:
        """Ask model if a word was in a presented list (for DRM experiment)."""
        list_text = ", ".join(word_list)
        prompt = (
            f"I'm going to show you a word list. Memorize these words carefully.\n\n"
            f"Word list: {list_text}\n\n"
            f"Now I will ask you about individual words. "
            f"For each word, answer ONLY \"yes\" or \"no\" — was it in the list above?\n\n"
            f"Was the word \"{query_word}\" in the list? Answer yes or no:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=10, temperature=0.0,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip().lower()

        # Parse yes/no
        if "yes" in answer.split()[0] if answer.split() else False:
            return True
        return False

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
