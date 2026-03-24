"""
Experiment D: Tip-of-Tongue (TOT) States

Tests whether architectures exhibit partial retrieval states where the
system "knows" the domain but fails on the specific answer.

For embedding architectures: TOT = target not rank-1 but in top-k (partial retrieval).
For LLM architectures: TOT = wrong answer but in correct domain/category.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional

from noescape.utils import set_seed, bootstrap_confidence_interval, load_wikipedia_sentences


def run_embedding_tot(architecture, config: dict, seed: int,
                       wiki_sentences: List[Dict] = None) -> dict:
    """
    TOT experiment for embedding-based architectures.

    TOT = target item not rank-1 but in top-k (ranks 2-20).
    This represents partial retrieval: the system finds the right neighborhood
    but not the exact item.
    """
    set_seed(seed)

    exp_cfg = config['experiments']['tot']
    n_queries = exp_cfg.get('n_queries', 1000)
    rank_range = exp_cfg.get('rank_range', [2, 20])
    noise_sigma = config['architectures'].get(architecture.arch_key, {}).get('noise_sigma', 0.25)

    if wiki_sentences is None:
        wiki_sentences = load_wikipedia_sentences(n_sentences=10000, n_articles=500)

    texts = [s['text'] for s in wiki_sentences[:min(n_queries * 10, len(wiki_sentences))]]

    if not texts:
        return {'seed': seed, 'tot_rate': 0, 'n_tot_states': 0, 'n_queries': 0}

    # Encode all
    embeddings = architecture.encode(texts)
    dim = embeddings.shape[1]

    n_tot = 0
    n_correct = 0
    n_total = 0

    for i in range(min(n_queries, len(texts))):
        target_emb = embeddings[i]

        # Add noise to simulate memory degradation
        age = np.random.uniform(1, 30)
        noise_scale = noise_sigma * np.sqrt(age + 0.01) / np.sqrt(dim)
        noisy_target = target_emb + noise_scale * np.random.randn(dim)
        noisy_target = noisy_target / (np.linalg.norm(noisy_target) + 1e-8)

        # Query against all memories
        sims = embeddings @ noisy_target
        ranks = np.argsort(sims)[::-1]
        target_rank = int(np.where(ranks == i)[0][0]) + 1  # 1-indexed

        n_total += 1
        if target_rank == 1:
            n_correct += 1
        elif rank_range[0] <= target_rank <= rank_range[1]:
            n_tot += 1

    tot_rate = n_tot / max(n_total, 1)

    return {
        'seed': seed,
        'tot_rate': float(tot_rate),
        'n_tot_states': n_tot,
        'n_queries': n_total,
        'n_correct': n_correct,
        'correct_rate': float(n_correct / max(n_total, 1)),
    }


def run_llm_tot(architecture, config: dict, seed: int,
                 wiki_sentences: List[Dict] = None) -> dict:
    """
    TOT experiment for LLM-based architectures.

    TOT = model gives wrong answer but demonstrates knowledge of the domain.
    """
    set_seed(seed)

    exp_cfg = config['experiments']['tot']
    n_queries = min(exp_cfg.get('n_queries', 1000), 200)  # Reduced for LLM speed

    if wiki_sentences is None:
        wiki_sentences = load_wikipedia_sentences(n_sentences=2000, n_articles=200)

    texts = [s['text'] for s in wiki_sentences[:min(n_queries * 5, len(wiki_sentences))]]

    # Create simple factual questions from sentences
    n_tot = 0
    n_correct = 0
    n_total = 0

    for i in range(min(n_queries, len(texts))):
        fact = texts[i]
        # Create a question about the fact
        question = f"What does this fact state: {fact[:60]}...?"

        # Store surrounding facts as context
        start = max(0, i - 50)
        end = min(len(texts), i + 50)
        context_facts = texts[start:end]

        if hasattr(architecture, 'answer_question_in_context'):
            answer = architecture.answer_question_in_context(context_facts, question)
        elif hasattr(architecture, 'answer_question'):
            answer = architecture.answer_question(question)
        else:
            architecture.clear()
            architecture.store(context_facts)
            results = architecture.retrieve(question, top_k=1)
            answer = architecture.get_item_text(results[0][0]) if results else ""

        # Check correctness
        fact_words = set(fact.lower().split())
        answer_words = set(answer.lower().split())
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'of', 'in', 'to', 'and', 'that', 'it', 'for'}
        fact_content = fact_words - stopwords
        answer_content = answer_words - stopwords

        if not fact_content:
            continue

        n_total += 1
        overlap = len(fact_content & answer_content) / len(fact_content)

        if overlap > 0.5:
            n_correct += 1
        elif overlap > 0.15:
            # Partial match = TOT state
            n_tot += 1

    tot_rate = n_tot / max(n_total, 1)

    return {
        'seed': seed,
        'tot_rate': float(tot_rate),
        'n_tot_states': n_tot,
        'n_queries': n_total,
        'n_correct': n_correct,
        'correct_rate': float(n_correct / max(n_total, 1)),
    }


def run_experiment(architecture, config: dict, seed: int,
                   wiki_sentences: List[Dict] = None) -> dict:
    if architecture.arch_key in ('vector_db', 'graph'):
        return run_embedding_tot(architecture, config, seed, wiki_sentences)
    else:
        return run_llm_tot(architecture, config, seed, wiki_sentences)


def run_all_seeds(architecture, config: dict, seeds: list = None,
                  wiki_sentences: List[Dict] = None) -> dict:
    if seeds is None:
        seeds = config['seeds']

    per_seed = {}
    for seed in seeds:
        print(f"Running TOT seed={seed} for {architecture.name}...")
        result = run_experiment(architecture, config, seed, wiki_sentences)
        per_seed[str(seed)] = result

    # Aggregate
    tot_rates = [per_seed[str(s)]['tot_rate'] for s in seeds]
    arr = np.array(tot_rates)
    ci = bootstrap_confidence_interval(arr) if len(arr) > 1 else (arr[0], arr[0])

    return {
        'architecture': architecture.arch_key,
        'experiment': 'tot',
        'per_seed': per_seed,
        'aggregated': {
            'tot_rate_mean': float(np.mean(arr)),
            'tot_rate_std': float(np.std(arr)),
            'tot_rate_ci': list(ci),
        },
    }


def save_results(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'tot.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved TOT results to {path}")
